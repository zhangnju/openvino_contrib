// Copyright (C) 2021-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <miopen/miopen.h>

#include <functional>
#include <optional>

#include "constant_factory.hpp"
#include "runtime.hpp"

inline std::string MIOPENGetErrorString(miopenConvFwdAlgorithm_t algo) {
    switch (algo) {
        case miopenConvolutionFwdAlgoGEMM:
            return "MIOPEN_CONVOLUTION_FWD_ALGO_GEMM";
        case miopenConvolutionFwdAlgoDirect:
            return "MIOPEN_CONVOLUTION_FWD_ALGO_DIRECT";
        case miopenConvolutionFwdAlgoFFT:
            return "MIOPEN_CONVOLUTION_FWD_ALGO_FFT";
        case miopenConvolutionFwdAlgoWinograd:
            return "MIOPEN_CONVOLUTION_FWD_ALGO_WINOGRAD";
        case miopenConvolutionFwdAlgoImplicitGEMM:
            return "MIOPEN_CONVOLUTION_FWD_ALGO_Implicit_GEMM";
        default:
            return "UNKNOWN MIOPEN_CONVOLUTION_ALGO";
    }
}

inline void throwIfError(
    miopenStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != miopenStatusSuccess) ov::rocm_gpu::throw_ov_exception(miopenGetErrorString(err), location);
}

inline void logIfError(
    miopenStatus_t err,
    const std::experimental::source_location& location = std::experimental::source_location::current()) {
    if (err != miopenStatusSuccess) ov::rocm_gpu::logError(miopenGetErrorString(err), location);
}

namespace rocm {
/*
class DnnOpTensorDescriptor : public Handle<miopenTensorDescriptor_t> {
public:
    DnnOpTensorDescriptor() : Handle((miopenCreateTensorDescriptor), miopenDestroyTensorDescriptor) {}
    auto& set(miopenTensorDescriptor_t opTensorOp,void *y, const void *alpha) {
        throwIfError(miopenSetTensor(get(), opTensorOp, y, alpha));
        return *this;
    }
};
*/
class DnnTensorDescriptor : public Handle<miopenTensorDescriptor_t> {
public:
    using CRef = std::reference_wrapper<const DnnTensorDescriptor>;

    DnnTensorDescriptor() : Handle((miopenCreateTensorDescriptor), miopenDestroyTensorDescriptor) {}

    auto& set(miopenDataType_t dataType, int nbDims, const int dimA[], const int strideA[]) {
        throwIfError(miopenSetTensorDescriptor(get(), dataType, nbDims, dimA, strideA));
        return *this;
    }

    auto& set(miopenTensorLayout_t format, miopenDataType_t dataType, int nbDims, const int dimA[]) {
        throwIfError(miopenSetNdTensorDescriptorWithLayout(get(), dataType, format, dimA, nbDims ));
        return *this;
    }

    auto& set(miopenTensorLayout_t format, miopenDataType_t dataType, int n, int c, int h, int w) {
        throwIfError(miopenSet4dTensorDescriptor(get(), dataType, n, c, h, w));
        return *this;
    }

    void getTensorNdDescriptor(int nbDimsRequested, miopenDataType_t& dataType, int& nbDims, int dimA[], int strideA[]) {
        //throwIfError(MIOPENGetTensorNdDescriptor(get(), nbDimsRequested, &dataType, &nbDims, dimA, strideA));
        throwIfError(miopenGetTensorDescriptor(get(), &dataType, dimA, strideA));
    }

    size_t getTensorSizeInBytes() {
        size_t size = 0;
        throwIfError(miopenGetTensorNumBytes(get(), &size));
        return size;
    }
};

class DnnActivationDescriptor : public Handle<miopenActivationDescriptor_t> {
public:
    DnnActivationDescriptor() : Handle((miopenCreateActivationDescriptor), miopenDestroyActivationDescriptor) {}
    auto& set(miopenActivationMode_t mode, double activAlpha, double activBeta, double activGamma) {
        throwIfError(miopenSetActivationDescriptor(get(), mode, activAlpha, activBeta,activGamma));
        return *this;
    }
};

class SigmoidDescriptor : public DnnActivationDescriptor {
public:
    SigmoidDescriptor() { set(miopenActivationLOGISTIC, 0, 0, 0); } //fix me, maybe need re-design
};

class ReluDescriptor : public DnnActivationDescriptor {
public:
    ReluDescriptor() : DnnActivationDescriptor{} { set(miopenActivationRELU, 0, 0, 0); } //fix me, maybe need re-design
};

class TanhDescriptor : public DnnActivationDescriptor {
public:
    TanhDescriptor() { set(miopenActivationTANH, 0, 0,0); } //fix me, maybe need re-design
};

class ClippedReluDescriptor : public DnnActivationDescriptor {
public:
    explicit ClippedReluDescriptor(double threshold) {
        set(miopenActivationCLIPPEDRELU, threshold, threshold, threshold); //fix me, maybe need re-design
    }
};

class DnnPoolingDescriptor : public Handle<miopenPoolingDescriptor_t> {
public:
    DnnPoolingDescriptor() : Handle((miopenCreatePoolingDescriptor), miopenDestroyPoolingDescriptor) {}
    auto& set(const miopenPoolingMode_t mode,
              const miopenNanPropagation_t nanPropagation,
              int nbDims,
              const int windowDimA[],
              const int paddingA[],
              const int strideA[]) {
        throwIfError(miopenSetNdPoolingDescriptor(get(), mode, nbDims, windowDimA, paddingA, strideA));
        return *this;
    }
};

class DnnFilterDescriptor : public Handle<miopenTensorDescriptor_t> {
public:
    DnnFilterDescriptor() : Handle((miopenCreateTensorDescriptor), miopenDestroyTensorDescriptor) {}
    auto& set(miopenDataType_t dataType, miopenTensorLayout_t format, int nbDims, const int filterDimA[]) {
        throwIfError(miopenSetNdTensorDescriptorWithLayout(get(), dataType, format, filterDimA, nbDims));
        return *this;
    }
};

class DnnRnnDataDescriptor : public Handle<miopenSeqTensorDescriptor_t> {
public:
    DnnRnnDataDescriptor() : Handle((miopenCreateSeqTensorDescriptor), miopenDestroySeqTensorDescriptor) {}
    auto& set(miopenDataType_t dataType,
              miopenRNNBaseLayout_t layout,
              int maxSeqLength,
              int batchSize,
              int vectorSize,
              const int seqLengthArray[],
              void* paddingFill) {
        throwIfError(miopenSetRNNDataSeqTensorDescriptor(
            get(), dataType, layout, maxSeqLength, batchSize, vectorSize, seqLengthArray, paddingFill));
        return *this;
    }
};
 
class DnnConvolutionDescriptor : public Handle<miopenConvolutionDescriptor_t> {
public:
    DnnConvolutionDescriptor() : Handle((miopenCreateConvolutionDescriptor), miopenDestroyConvolutionDescriptor) {}
    auto& set(int arrayLength,
              const int padA[],
              const int filterStrideA[],
              const int dilationA[],
              miopenConvolutionMode_t mode,
              const int groups) {
        throwIfError(
            miopenInitConvolutionNdDescriptor(get(), arrayLength, padA, filterStrideA, dilationA, mode)
            //miopenInitConvolutionDescriptor(get(),mode,0, 0, 1, 1, 1, 1)
            );
        throwIfError(
            miopenSetConvolutionGroupCount(get(), groups)
        );
        return *this;
    }
};

class DnnRnnDescriptor : public Handle<miopenRNNDescriptor_t> {
public:
    DnnRnnDescriptor() : Handle((miopenCreateRNNDescriptor), miopenDestroyRNNDescriptor) {}
    auto& set(miopenRNNAlgo_t algo,
              miopenRNNMode_t cellMode,
              miopenRNNBiasMode_t biasMode,
              miopenRNNDirectionMode_t dirMode,
              miopenRNNInputMode_t inputMode,
              miopenDataType_t dataType,
              int32_t inputSize,
              int32_t hiddenSize,
              int32_t numLayers,
              miopenDropoutDescriptor_t dropoutDesc
              ) {
        throwIfError(miopenSetRNNDescriptor_V2(get(),
                                              hiddenSize,
                                              numLayers,
                                              dropoutDesc,
                                              inputMode,
                                              dirMode,
                                              cellMode,
                                              biasMode,
                                              algo,
                                              dataType
                                              ));
        return *this;
    }
    /*
    auto& setClip(miopenRNNClipMode_t clipMode, miopenNanPropagation_t clipNanOpt, double lclip, double rclip) {
        throwIfError(miopenRNNSetClip_v8(get(), clipMode, clipNanOpt, lclip, rclip));
        return *this;
    }
    */
};

class DnnReduceTensorDescriptor : public Handle<miopenReduceTensorDescriptor_t> {
public:
    DnnReduceTensorDescriptor()
        : Handle<miopenReduceTensorDescriptor_t>((miopenCreateReduceTensorDescriptor),
                                                miopenDestroyReduceTensorDescriptor) {}
    auto& set(miopenReduceTensorOp_t op,
              miopenDataType_t compType,
              miopenNanPropagation_t nanOpt,
              miopenReduceTensorIndices_t indices,
              miopenIndicesType_t indicesType) {
        throwIfError(miopenSetReduceTensorDescriptor(get(), op, compType, nanOpt, indices, indicesType));
        return *this;
    }
};

class DnnReduceAddDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceAddDescriptor(miopenDataType_t compType) {
        set(MIOPEN_REDUCE_TENSOR_ADD,
            compType,
            MIOPEN_PROPAGATE_NAN,
            MIOPEN_REDUCE_TENSOR_NO_INDICES,
            MIOPEN_32BIT_INDICES);
    }
};

class DnnReduceMulDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceMulDescriptor(miopenDataType_t compType) {
        set(MIOPEN_REDUCE_TENSOR_MUL,
            compType,
            MIOPEN_PROPAGATE_NAN,
            MIOPEN_REDUCE_TENSOR_NO_INDICES,
            MIOPEN_32BIT_INDICES);
    }
};

class DnnReduceMinDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceMinDescriptor(miopenDataType_t compType) {
        set(MIOPEN_REDUCE_TENSOR_MIN,
            compType,
            MIOPEN_PROPAGATE_NAN,
            MIOPEN_REDUCE_TENSOR_NO_INDICES,
            MIOPEN_32BIT_INDICES);
    }
};

class DnnReduceMaxDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceMaxDescriptor(miopenDataType_t compType) {
        set(MIOPEN_REDUCE_TENSOR_MAX,
            compType,
            MIOPEN_PROPAGATE_NAN,
            MIOPEN_REDUCE_TENSOR_NO_INDICES,
            MIOPEN_32BIT_INDICES);
    }
};

class DnnReduceAvgDescriptor : public DnnReduceTensorDescriptor {
public:
    explicit DnnReduceAvgDescriptor(miopenDataType_t compType) {
        set(MIOPEN_REDUCE_TENSOR_AVG,
            compType,
            MIOPEN_PROPAGATE_NAN,
            MIOPEN_REDUCE_TENSOR_NO_INDICES,
            MIOPEN_32BIT_INDICES);
    }
};

class DnnScaleFactor {
public:
    constexpr const void* get() const noexcept { return scaling_factor_; }

protected:
    explicit constexpr DnnScaleFactor(const void* scalingFactor) noexcept : scaling_factor_{scalingFactor} {}

private:
    const void* scaling_factor_;
};

class DnnScaleFactorZero : public DnnScaleFactor {
public:
    explicit constexpr DnnScaleFactorZero(miopenDataType_t compType) noexcept
        : DnnScaleFactor{&rocm::NumericConst<rocm::constants::zero>(compType)} {}
};

class DnnScaleFactorOne : public DnnScaleFactor {
public:
    explicit constexpr DnnScaleFactorOne(miopenDataType_t compType) noexcept
        : DnnScaleFactor{&rocm::NumericConst<rocm::constants::one>(compType)} {}
};

class DnnHandle : public Handle<miopenHandle_t> {
public:
    DnnHandle() : Handle((miopenCreate), miopenDestroy) {}
    void setStream(const Stream& stream) { throwIfError(miopenSetStream(get(), stream.get())); }
    auto getStream() const {
        hipStream_t stream = nullptr;
        throwIfError(::miopenGetStream(get(), &stream));
        return stream;
    }
    void opTensor(const miopenTensorOp_t tensorOp,
                  const void* alpha1,
                  const DnnTensorDescriptor& aDesc,
                  const void* A,
                  const void* alpha2,
                  const DnnTensorDescriptor& bDesc,
                  const void* B,
                  const void* beta,
                  const DnnTensorDescriptor& cDesc,
                  void* C) const {
        throwIfError(miopenOpTensor(
            get(), tensorOp, alpha1, aDesc.get(), A, alpha2, bDesc.get(), B, beta, cDesc.get(), C));
    }
    // TODO: accept device pointers for x and y
    void activationForward(const DnnActivationDescriptor& activationDesc,
                           const void* alpha,
                           const DnnTensorDescriptor& xDesc,
                           const void* x,
                           const void* beta,
                           const DnnTensorDescriptor& yDesc,
                           void* y) const {
        throwIfError(miopenActivationForward(get(), activationDesc.get(), alpha, xDesc.get(), x, beta, yDesc.get(), y));
    }
    void rnnForward(const DnnRnnDescriptor& rnnDesc,
                    miopenRNNFWDMode_t fwdMode,
                    const int32_t devSeqLengths[],
                    const DnnRnnDataDescriptor& xDesc,
                    const void* x,
                    const DnnRnnDataDescriptor& yDesc,
                    void* y,
                    const DnnTensorDescriptor& hDesc,
                    const void* hx,
                    void* hy,
                    std::optional<DnnTensorDescriptor::CRef> cDesc,
                    const void* cx,
                    void* cy,
                    size_t weightSpaceSize,
                    const void* weightSpace,
                    size_t workSpaceSize,
                    void* workSpace,
                    size_t reserveSpaceSize,
                    void* reserveSpace) const {
        throwIfError(miopenRNNForward(get(),
                                     rnnDesc.get(),
                                     fwdMode,
                                     xDesc.get(),
                                     x,
                                     hDesc.get(),
                                     hx,
                                     hy,
                                     cDesc ? cDesc->get().get() : nullptr,
                                     cx,
                                     cy,
                                     yDesc.get(),
                                     y,
                                     weightSpace,
                                     weightSpaceSize,
                                     workSpace,
                                     workSpaceSize,
                                     reserveSpace,
                                     reserveSpaceSize
                                     ));
    }
    size_t getReductionWorkspaceSize(const DnnReduceTensorDescriptor& reduceDesc,
                                     const DnnTensorDescriptor& aDesc,
                                     const DnnTensorDescriptor& cDesc) const {
        return createLastArg(miopenGetReductionWorkspaceSize, get(), reduceDesc, aDesc, cDesc);
    }
    void reduceTensor(const DnnReduceTensorDescriptor& reduceTensorDesc,
                      rocm::DeviceBuffer<std::uint8_t> workspace,
                      const DnnScaleFactor& alpha,
                      const DnnTensorDescriptor& aDesc,
                      rocm::DevicePointer<const void*> a,
                      const DnnScaleFactor& beta,
                      const DnnTensorDescriptor& cDesc,
                      rocm::DevicePointer<void*> c) const {
        throwIfError(miopenReduceTensor(get(),
                                       reduceTensorDesc.get(),
                                       nullptr,
                                       0,
                                       workspace.data(),
                                       workspace.size_bytes(),
                                       alpha.get(),
                                       aDesc.get(),
                                       a.get(),
                                       beta.get(),
                                       cDesc.get(),
                                       c.get()));
    }
};

}  // namespace rocm
