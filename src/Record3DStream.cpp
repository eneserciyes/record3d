#include "../include/record3d/Record3DStream.h"
#include <usbmuxd.h>
#include <cstring>
#include <array>
#include <string>

#define NTOHL_(n) (((((unsigned long)(n) & 0xFF)) << 24) | \
                  ((((unsigned long)(n) & 0xFF00)) << 8) | \
                  ((((unsigned long)(n) & 0xFF0000)) >> 8) | \
                  ((((unsigned long)(n) & 0xFF000000)) >> 24))

/// The public part
namespace Record3D
{
    Record3DStream::Record3DStream()
    {
    }

    Record3DStream::~Record3DStream()
    {
    }

    std::vector<DeviceInfo> Record3DStream::GetConnectedDevices()
    {
        usbmuxd_device_info_t* deviceInfoList;
        int numDevices = usbmuxd_get_device_list( &deviceInfoList );
        std::vector<DeviceInfo> availableDevices;

        for ( int devIdx = 0; devIdx < numDevices; devIdx++ )
        {
            const auto &dev = deviceInfoList[ devIdx ];
            if ( dev.conn_type != CONNECTION_TYPE_USB ) continue;

            DeviceInfo currDevInfo;
            currDevInfo.handle = dev.handle;
            currDevInfo.productId = dev.product_id;
            currDevInfo.udid = std::string( dev.udid );

            availableDevices.push_back( currDevInfo );
        }

        usbmuxd_device_list_free( &deviceInfoList );

        return availableDevices;
    }


    bool Record3DStream::ConnectToDevice(const DeviceInfo &$device)
    {
        std::lock_guard<std::mutex> guard{ apiCallsMutex_ };

        // Do not reconnect if we are already streaming.
        if ( connectionEstablished_.load())
        { return false; }

        // Ensure we are indeed connected before continuing.
        auto socketNo = usbmuxd_connect( $device.handle, DEVICE_PORT );
        if ( socketNo < 0 )
        { return false; }

        // We are successfully connected, start runloop.
        connectionEstablished_.store( true );
        socketHandle_ = socketNo;

        // Create thread that is going to execute runloop.
        runloopThread_ = std::thread( [&]
                                      {
                                          StreamProcessingRunloop();
                                      } );
        runloopThread_.detach();
        return true;
    }

    void Record3DStream::Disconnect()
    {
        std::lock_guard<std::mutex> guard{ apiCallsMutex_ };

        connectionEstablished_.store( false );

        if ( onStreamStopped )
        {
            onStreamStopped();
        }
    }
}


/// The private part
namespace Record3D
{
    struct PeerTalkHeader
    {
        uint32_t a;
        uint32_t b;
        uint32_t c;
        uint32_t body_size;
    };

    struct Record3DHeader
    {
        uint32_t rgbWidth;
        uint32_t rgbHeight;
        uint32_t depthWidth;
        uint32_t depthHeight;
        uint32_t confidenceWidth;
        uint32_t confidenceHeight;
        uint32_t rgbSize;
        uint32_t depthSize;
        uint32_t confidenceMapSize;
        uint32_t miscSize;
        uint32_t deviceType;
    };

    void Record3DStream::StreamProcessingRunloop()
    {
        std::vector<uint8_t> rawMessageBuffer;
        uint32_t numReceivedData = 0;

        while ( connectionEstablished_.load())
        {
            // 1. Receive the PeerTalk header
            PeerTalkHeader ptHeader;
            numReceivedData = ReceiveWholeBuffer( socketHandle_, (uint8_t*) &ptHeader, sizeof( ptHeader ));
            uint32_t messageBodySize = NTOHL_( ptHeader.body_size );

            if ( numReceivedData != sizeof( ptHeader ))
            { break; }

            // 2. Receive the whole body
            if ( rawMessageBuffer.size() < messageBodySize )
            {
                rawMessageBuffer.resize( messageBodySize );
            }

            numReceivedData = ReceiveWholeBuffer( socketHandle_, (uint8_t*) rawMessageBuffer.data(),
                                                  messageBodySize );
            if ( numReceivedData != messageBodySize )
            { break; }

            // 3. Parse the body
            Record3DHeader record3DHeader;

            size_t offset = 0;
            size_t currSize = 0;

            // 3.1 Read the header of Record3D
            currSize = sizeof( Record3DHeader );
            memcpy((void*) &record3DHeader, rawMessageBuffer.data() + offset, currSize );
            currentDeviceType_ = (DeviceType)record3DHeader.deviceType;
            offset += currSize;

            // 3.2 Read intrinsic matrix coefficients
            currSize = sizeof( IntrinsicMatrixCoeffs );
            memcpy((void*) &rgbIntrinsicMatrixCoeffs_, rawMessageBuffer.data() + offset, currSize );
            offset += currSize;

            // 3.3 Read the camera pose data
            currSize = sizeof( CameraPose );
            memcpy( (void*) &cameraPose_, rawMessageBuffer.data() + offset, currSize );
            offset += currSize;

            // 3.3 Read and decode the RGB frame
            currSize = record3DHeader.rgbSize;
            if ( RGBImageBuffer_.size() != currSize )
            {
                RGBImageBuffer_.resize(currSize);
            }
            memcpy( RGBImageBuffer_.data(), rawMessageBuffer.data() + offset, currSize);
            offset += currSize;

            // 3.4 Read and decompress the depth frame
            currSize = record3DHeader.depthSize;
            // Resize the decompressed depth image buffer
            if ( depthImageBuffer_.size() != currSize )
            {
                depthImageBuffer_.resize(currSize);
            }
            memcpy( depthImageBuffer_.data(), rawMessageBuffer.data() + offset, currSize );
            offset += currSize;

            // 3.5 Read and decompress the confidence frame corresponding to the depth frame
            currSize = record3DHeader.confidenceMapSize;
            // Resize the decompressed confidence image buffer
            if ( confidenceImageBuffer_.size() != currSize )
            {
                confidenceImageBuffer_.resize(currSize);
            }

            memcpy( confidenceImageBuffer_.data(), rawMessageBuffer.data() + offset, currSize );

            offset += currSize;

            // 3.6 Read the misc buffer
            if ( record3DHeader.miscSize > 0 )
            {
                currSize = record3DHeader.miscSize;

                miscBuffer_.resize( currSize );
                memcpy(miscBuffer_.data(), rawMessageBuffer.data() + offset, currSize );

                offset += currSize;
            }

            if ( onNewFrame )
            {
                currentFrameRGBWidth_ = record3DHeader.rgbWidth;
                currentFrameRGBHeight_ = record3DHeader.rgbHeight;

                currentFrameDepthWidth_ = record3DHeader.depthWidth;
                currentFrameDepthHeight_ = record3DHeader.depthHeight;

                currentFrameConfidenceWidth_ = record3DHeader.confidenceWidth;
                currentFrameConfidenceHeight_ = record3DHeader.confidenceHeight;

#ifdef PYTHON_BINDINGS_BUILD
                onNewFrame( );
#else
                onNewFrame( RGBImageBuffer_,
                            depthImageBuffer_,
                            confidenceImageBuffer_,
                            miscBuffer_,
                            record3DHeader.rgbWidth,
                            record3DHeader.rgbHeight,
                            record3DHeader.depthWidth,
                            record3DHeader.depthHeight,
                            record3DHeader.confidenceWidth,
                            record3DHeader.confidenceHeight,
                            currentDeviceType_,
                            rgbIntrinsicMatrixCoeffs_,
                            cameraPose_ );
#endif
            }
        }

        Disconnect();
    }

    uint32_t Record3DStream::ReceiveWholeBuffer(int $socketHandle, uint8_t* $outputBuffer, uint32_t $numBytesToRead)
    {
        uint32_t numTotalReceivedBytes = 0;
        while ( numTotalReceivedBytes < $numBytesToRead )
        {
            uint32_t numRestBytes = $numBytesToRead - numTotalReceivedBytes;
            uint32_t numActuallyReceivedBytes = 0;
            if ( 0 != usbmuxd_recv( $socketHandle, (char*) ($outputBuffer + numTotalReceivedBytes), numRestBytes,
                                    &numActuallyReceivedBytes ))
            {
#if DEBUG
                fprintf( stderr, "ERROR WHILE RECEIVING DATA!\n" );
#endif
                return numTotalReceivedBytes;
            }
            numTotalReceivedBytes += numActuallyReceivedBytes;
        }

        return numTotalReceivedBytes;
    }
}
