#ifndef KALDI_NNET_GRU_STREAMS_H_
#define KALDI_NNET_GRU_STREAMS_H_

#include "nnet/nnet-component.h"
#include "nnet/nnet-utils.h"
#include "cudamatrix/cu-math.h"

/*************************************
 * x: input neuron
 *************************************/

namespace kaldi {
namespace nnet1 {
class GruStreams : public UpdatableComponent {
public:
    GruStreams(int32 input_dim, int32 output_dim) :
        UpdatableComponent(input_dim, output_dim),
        nstream_(0),
        clip_gradient_(0.0)
        //, dropout_rate_(0.0)
    { }

    ~GruStreams()
    { }

    Component* Copy() const { return new GruStreams(*this); }
    ComponentType GetType() const { return kGruStreams; }

    static void InitMatParam(CuMatrix<BaseFloat> &m, float scale) {
        m.SetRandUniform();  // uniform in [0, 1]
        m.Add(-0.5);         // uniform in [-0.5, 0.5]
        m.Scale(2 * scale);  // uniform in [-scale, +scale]
    }

    static void InitVecParam(CuVector<BaseFloat> &v, float scale) {
        Vector<BaseFloat> tmp(v.Dim());
        for (int i=0; i < tmp.Dim(); i++) {
            tmp(i) = (RandUniform() - 0.5) * 2 * scale;
        }
        v = tmp;
    }

    void InitData(std::istream &is) {
        // define options
        float param_scale = 0.02;
        // parse config
        std::string token;
        while (!is.eof()) {
            ReadToken(is, false, &token); 
            if (token == "<ClipGradient>") 
                ReadBasicType(is, false, &clip_gradient_);
            //else if (token == "<DropoutRate>") 
            //    ReadBasicType(is, false, &dropout_rate_);
            else if (token == "<ParamScale>") 
                ReadBasicType(is, false, &param_scale);
            else KALDI_ERR << "Unknown token " << token << ", a typo in config?"
                           << " (CellDim|NumStream|ParamScale)";
                           //<< " (CellDim|NumStream|DropoutRate|ParamScale)";
            is >> std::ws;
        }

        // init weight and bias (Uniform)
        w_z_x_.Resize(output_dim_, input_dim_, kUndefined);
        w_r_x_.Resize(output_dim_, input_dim_, kUndefined);
        w_z_h_.Resize(output_dim_, output_dim_, kUndefined);
        w_r_h_.Resize(output_dim_, output_dim_, kUndefined);
        w_m_x_.Resize(output_dim_, input_dim_, kUndefined);
        w_m_g_.Resize(output_dim_, output_dim_, kUndefined);

        InitMatParam(w_z_x_, param_scale);
        InitMatParam(w_r_x_, param_scale);
        InitMatParam(w_z_h_, param_scale);
        InitMatParam(w_r_h_, param_scale);
        InitMatParam(w_m_x_, param_scale);
        InitMatParam(w_m_g_, param_scale);

        // init delta buffers
        w_z_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_r_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_z_h_corr_.Resize(output_dim_, output_dim_, kSetZero); 
        w_r_h_corr_.Resize(output_dim_, output_dim_, kSetZero); 
        w_m_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_m_g_corr_.Resize(output_dim_, output_dim_, kSetZero); 

        KALDI_ASSERT(clip_gradient_ >= 0.0);
    }

    void ReadData(std::istream &is, bool binary) {
        ExpectToken(is, binary, "<ClipGradient>");
        ReadBasicType(is, binary, &clip_gradient_);
        //ExpectToken(is, binary, "<DropoutRate>");
        //ReadBasicType(is, binary, &dropout_rate_);
        
        w_z_x_.Read(is, binary);
        w_r_x_.Read(is, binary);
        w_z_h_.Read(is, binary);
        w_r_h_.Read(is, binary);
        w_m_x_.Read(is, binary);
        w_m_g_.Read(is, binary);

        w_z_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_r_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_z_h_corr_.Resize(output_dim_, output_dim_, kSetZero); 
        w_r_h_corr_.Resize(output_dim_, output_dim_, kSetZero); 
        w_m_x_corr_.Resize(output_dim_, input_dim_, kSetZero); 
        w_m_g_corr_.Resize(output_dim_, output_dim_, kSetZero); 
    }

    void WriteData(std::ostream &os, bool binary) const {
        WriteToken(os, binary, "<ClipGradient>");
        WriteBasicType(os, binary, clip_gradient_);
        //WriteToken(os, binary, "<DropoutRate>");
        //WriteBasicType(os, binary, dropout_rate_);
        w_z_x_.Write(os, binary);
        w_r_x_.Write(os, binary);
        w_z_h_.Write(os, binary);
        w_r_h_.Write(os, binary);
        w_m_x_.Write(os, binary);
        w_m_g_.Write(os, binary);
    }

    int32 NumParams() const { 
        return ( w_z_x_.NumRows() * w_z_x_.NumCols() +
                 w_r_x_.NumRows() * w_r_x_.NumCols() +
                 w_z_h_.NumRows() * w_z_h_.NumCols() +
                 w_r_h_.NumRows() * w_r_h_.NumCols() +
                 w_m_x_.NumRows() * w_m_x_.NumCols() +
                 w_m_g_.NumRows() * w_m_g_.NumCols() );
    }

    void GetParams(Vector<BaseFloat>* wei_copy) const {
        wei_copy->Resize(NumParams());

        int32 offset, len;

        offset = 0;    len = w_z_x_.NumRows() * w_z_x_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_z_x_);

        offset += len; len = w_r_x_.NumRows() * w_r_x_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_r_x_);

        offset += len; len = w_z_h_.NumRows() * w_z_h_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_z_h_);

        offset += len; len = w_r_h_.NumRows() * w_r_h_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_r_h_);

        offset += len; len = w_m_x_.NumRows() * w_m_x_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_m_x_);

        offset += len; len = w_m_g_.NumRows() * w_m_g_.NumCols();
        wei_copy->Range(offset, len).CopyRowsFromMat(w_m_g_);

        return;
    }

    std::string Info() const {
        return std::string("    ") + 
            "\n  w_z_x_  "     + MomentStatistics(w_z_x_) + 
            "\n  w_r_x_  "     + MomentStatistics(w_r_x_) +
            "\n  w_z_h_  "     + MomentStatistics(w_z_h_) +
            "\n  w_r_h_  "     + MomentStatistics(w_r_h_) +
            "\n  w_m_x_  "     + MomentStatistics(w_m_x_) +
            "\n  w_m_g_  "     + MomentStatistics(w_m_g_);
    }
  
    std::string InfoGradient() const {
        // disassemble forward-propagation buffer into different neurons,
        const CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(2*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(3*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));

        // disassemble backpropagate buffer into different neurons,
        const CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(0*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(1*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(2*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(3*output_dim_, output_dim_));
        const CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(4*output_dim_, output_dim_));

        return std::string("    ") + 
            "\n  Gradients:" +
            "\n    w_z_x_corr_  "     + MomentStatistics(w_z_x_corr_) + 
            "\n    w_r_x_corr_  "     + MomentStatistics(w_r_x_corr_) +
            "\n    w_z_h_corr_  "     + MomentStatistics(w_z_h_corr_) +
            "\n    w_r_h_corr_  "     + MomentStatistics(w_r_h_corr_) +
            "\n    w_m_x_corr_  "     + MomentStatistics(w_m_x_corr_) +
            "\n    w_m_g_corr_  "     + MomentStatistics(w_m_g_corr_) +
            "\n  Forward-pass:" +
            "\n    YZ  " + MomentStatistics(YZ) +
            "\n    YR  " + MomentStatistics(YR) +
            "\n    YG  " + MomentStatistics(YG) +
            "\n    YM  " + MomentStatistics(YM) +
            "\n    YH  " + MomentStatistics(YH) +
            "\n  Backward-pass:" +
            "\n    DZ  " + MomentStatistics(DZ) +
            "\n    DR  " + MomentStatistics(DR) +
            "\n    DG  " + MomentStatistics(DG) +
            "\n    DM  " + MomentStatistics(DM) +
            "\n    DH  " + MomentStatistics(DH);
    }

    void ResetLstmStreams(const std::vector<int32> &stream_reset_flag) {
        // allocate prev_nnet_state_ if not done yet,
        if (nstream_ == 0) {
          // Karel: we just got number of streams! (before the 1st batch comes)
          nstream_ = stream_reset_flag.size(); 
          prev_nnet_state_.Resize(nstream_, 5 * output_dim_ , kSetZero);
          KALDI_LOG << "Running training with " << nstream_ << " streams.";
        }
        // reset flag: 1 - reset stream network state
        KALDI_ASSERT(prev_nnet_state_.NumRows() == stream_reset_flag.size());
        for (int s = 0; s < stream_reset_flag.size(); s++) {
            if (stream_reset_flag[s] == 1) {
                prev_nnet_state_.Row(s).SetZero();
            }
        }
    }

    void PropagateFnc(const CuMatrixBase<BaseFloat> &in, CuMatrixBase<BaseFloat> *out) {
        int DEBUG = 0;

        static bool do_stream_reset = false;
        if (nstream_ == 0) {
          do_stream_reset = true;
          nstream_ = 1; // Karel: we are in nnet-forward, so 1 stream,
          prev_nnet_state_.Resize(nstream_, 5 * output_dim_, kSetZero);
          KALDI_LOG << "Running nnet-forward with per-utterance LSTM-state reset";
        }
        if (do_stream_reset) prev_nnet_state_.SetZero();
        KALDI_ASSERT(nstream_ > 0);

        KALDI_ASSERT(in.NumRows() % nstream_ == 0);
        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;

        // 0:forward pass history, [1, T]:current sequence, T+1:dummy
        propagate_buf_.Resize((T+2)*S, 5 * output_dim_, kSetZero);  
        propagate_buf_.RowRange(0*S,S).CopyFromMat(prev_nnet_state_);

        // disassemble entire neuron activation buffer into different neurons
        CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(2*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(3*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));

        // x -> z, r, m, not recurrent, do it all in once
        YZ.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_z_x_, kTrans, 0.0);
        YR.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_r_x_, kTrans, 0.0);
        YM.RowRange(1*S,T*S).AddMatMat(1.0, in, kNoTrans, w_m_x_, kTrans, 0.0);

        for (int t = 1; t <= T; t++) {
            // multistream buffers for current time-step
            CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));  
    
            // h(t-1) -> z, r
            y_z.AddMatMat(1.0, YH.RowRange((t-1)*S,S), kNoTrans, w_z_h_, kTrans,  1.0);
            y_r.AddMatMat(1.0, YH.RowRange((t-1)*S,S), kNoTrans, w_r_h_, kTrans,  1.0);

            // z, r sigmoid squashing
            y_z.Sigmoid(y_z);
            y_r.Sigmoid(y_r);

            // r(t) * h(t-1) -> g(t)
            y_g.AddMatMatElements(1.0, YH.RowRange((t-1)*S,S), y_r, 0.0);
           
            // g(t) -> m(t)
            y_m.AddMatMat(1.0, y_g, kNoTrans, w_m_g_, kTrans, 1.0);

            // m tanh squashing
            y_m.Tanh(y_m);
   
            // h(t-1), m(t), z(t) -> h(t)
            y_h.AddMat(1.0, YH.RowRange((t-1)*S,S));
            y_h.AddMatMatElements(-1.0, YH.RowRange((t-1)*S,S), y_z, 1.0);
            y_h.AddMatMatElements(1.0, y_m, y_z, 1.0);

            if (DEBUG) {
                std::cerr << "forward-pass frame " << t << "\n";
                std::cerr << "activation of z: " << y_z;
                std::cerr << "activation of r: " << y_r;
                std::cerr << "activation of g: " << y_g;
                std::cerr << "activation of m: " << y_m;
                std::cerr << "activation of h: " << y_h;
            }
        }

        // h is feed-forward as GRU output
        out->CopyFromMat(YH.RowRange(1*S,T*S));

        // now the last frame state becomes previous network state for next batch
        prev_nnet_state_.CopyFromMat(propagate_buf_.RowRange(T*S,S));
    }

    void BackpropagateFnc(const CuMatrixBase<BaseFloat> &in, const CuMatrixBase<BaseFloat> &out,
                          const CuMatrixBase<BaseFloat> &out_diff, CuMatrixBase<BaseFloat> *in_diff) {

        int DEBUG = 0;

        int32 T = in.NumRows() / nstream_;
        int32 S = nstream_;

        // disassemble propagated buffer into neurons
        CuSubMatrix<BaseFloat> YZ(propagate_buf_.ColRange(0*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YR(propagate_buf_.ColRange(1*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YG(propagate_buf_.ColRange(2*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YM(propagate_buf_.ColRange(3*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> YH(propagate_buf_.ColRange(4*output_dim_, output_dim_));
    
        // 0:dummy, [1,T] frames, T+1 backward pass history
        backpropagate_buf_.Resize((T+2)*S, 5 * output_dim_, kSetZero);

        // disassemble backpropagate buffer into neurons
        CuSubMatrix<BaseFloat> DZ(backpropagate_buf_.ColRange(0*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> DR(backpropagate_buf_.ColRange(1*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> DG(backpropagate_buf_.ColRange(2*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> DM(backpropagate_buf_.ColRange(3*output_dim_, output_dim_));
        CuSubMatrix<BaseFloat> DH(backpropagate_buf_.ColRange(4*output_dim_, output_dim_));

        // projection layer to LSTM output is not recurrent, so backprop it all in once
        DH.RowRange(1*S,T*S).CopyFromMat(out_diff);

        for (int t = T; t >= 1; t--) {
            CuSubMatrix<BaseFloat> y_z(YZ.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_r(YR.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_g(YG.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_m(YM.RowRange(t*S,S));  
            CuSubMatrix<BaseFloat> y_h(YH.RowRange(t*S,S));  
    
            CuSubMatrix<BaseFloat> d_z(DZ.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_r(DR.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_g(DG.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_m(DM.RowRange(t*S,S));
            CuSubMatrix<BaseFloat> d_h(DH.RowRange(t*S,S));
            
            // h
            d_h.AddMatMat(1.0, DR.RowRange((t+1)*S,S), kNoTrans, w_r_h_, kNoTrans, 1.0);
            d_h.AddMatMat(1.0, DZ.RowRange((t+1)*S,S), kNoTrans, w_z_h_, kNoTrans, 1.0);
            d_h.AddMat(1.0, DH.RowRange((t+1)*S,S));
            d_h.AddMatMatElements(-1.0, DH.RowRange((t+1)*S,S), YZ.RowRange((t+1)*S,S), 1.0);
            d_h.AddMatMatElements(1.0, DG.RowRange((t+1)*S,S), YR.RowRange((t+1)*S,S), 1.0);
            
            // m
            d_m.AddMatMatElements(1.0, d_h, y_z, 0.0);
            d_m.DiffTanh(y_m, d_m);
            
            // g
            d_g.AddMatMat(1.0, d_m, kNoTrans, w_m_g_, kNoTrans, 0.0);

            // r
            d_r.AddMatMatElements(1.0, d_g, YH.RowRange((t-1)*S,S), 0.0);
            d_r.DiffSigmoid(y_r, d_r);

            // z
            d_z.AddMatMatElements( 1.0, d_h, y_m, 0.0);
            d_z.AddMatMatElements(-1.0, d_h, YH.RowRange((t-1)*S,S), 1.0);
            d_z.DiffSigmoid(y_z, d_z);
            
            // debug info
            if (DEBUG) {
                std::cerr << "backward-pass frame " << t << "\n";
                std::cerr << "derivative wrt input r " << d_z;
                std::cerr << "derivative wrt input m " << d_r;
                std::cerr << "derivative wrt input h " << d_g;
                std::cerr << "derivative wrt input o " << d_m;
                std::cerr << "derivative wrt input c " << d_h;
            }
        }
        
        // z,r,m -> x, do it all in once
        in_diff->AddMatMat(1.0, DZ.RowRange(1*S,T*S), kNoTrans, w_z_x_, kNoTrans, 0.0);
        in_diff->AddMatMat(1.0, DR.RowRange(1*S,T*S), kNoTrans, w_r_x_, kNoTrans, 1.0);
        in_diff->AddMatMat(1.0, DM.RowRange(1*S,T*S), kNoTrans, w_m_x_, kNoTrans, 1.0);
        //// backward pass dropout
        //if (dropout_rate_ != 0.0) {
        //    in_diff->MulElements(dropout_mask_);
        //}
    
        // calculate delta
        const BaseFloat mmt = opts_.momentum;
    
        // weight x -> z
        w_z_x_corr_.AddMatMat(1.0, DZ.RowRange(1*S,T*S), kTrans, 
                                   in                  , kNoTrans, mmt);
        // weight x -> r
        w_r_x_corr_.AddMatMat(1.0, DR.RowRange(1*S,T*S), kTrans, 
                                   in                  , kNoTrans, mmt);
        // weight x -> m
        w_m_x_corr_.AddMatMat(1.0, DM.RowRange(1*S,T*S), kTrans, 
                                   in                  , kNoTrans, mmt);
        // recurrent weight h -> z, r
        w_z_h_corr_.AddMatMat(1.0, DZ.RowRange(1*S,T*S), kTrans, 
                                   YH.RowRange(0*S,T*S), kNoTrans, mmt);
        w_r_h_corr_.AddMatMat(1.0, DR.RowRange(1*S,T*S), kTrans, 
                                   YH.RowRange(0*S,T*S), kNoTrans, mmt);
        // recurrent weight g -> m
        w_m_g_corr_.AddMatMat(1.0, DM.RowRange(1*S,T*S), kTrans, 
                                   YG.RowRange(1*S,T*S), kNoTrans, mmt);

        if (clip_gradient_ > 0.0) {
          w_z_x_corr_.ApplyFloor(-clip_gradient_);
          w_z_x_corr_.ApplyCeiling(clip_gradient_);
          w_r_x_corr_.ApplyFloor(-clip_gradient_);
          w_r_x_corr_.ApplyCeiling(clip_gradient_);
          w_m_x_corr_.ApplyFloor(-clip_gradient_);
          w_m_x_corr_.ApplyCeiling(clip_gradient_);
          w_z_h_corr_.ApplyFloor(-clip_gradient_);
          w_z_h_corr_.ApplyCeiling(clip_gradient_);
          w_r_h_corr_.ApplyFloor(-clip_gradient_);
          w_r_h_corr_.ApplyCeiling(clip_gradient_);
          w_m_g_corr_.ApplyFloor(-clip_gradient_);
          w_m_g_corr_.ApplyCeiling(clip_gradient_);
        }

        if (DEBUG) {
            std::cerr << "gradients(with optional momentum): \n";
            std::cerr << "w_gifo_x_corr_ " << w_gifo_x_corr_;
            std::cerr << "w_gifo_r_corr_ " << w_gifo_r_corr_;
            std::cerr << "bias_corr_ " << bias_corr_;
            std::cerr << "w_r_m_corr_ " << w_r_m_corr_;
            std::cerr << "peephole_i_c_corr_ " << peephole_i_c_corr_;
            std::cerr << "peephole_f_c_corr_ " << peephole_f_c_corr_;
            std::cerr << "peephole_o_c_corr_ " << peephole_o_c_corr_;
        }
    }

    void Update(const CuMatrixBase<BaseFloat> &input, const CuMatrixBase<BaseFloat> &diff) {
        const BaseFloat lr  = opts_.learn_rate;

        w_gifo_x_.AddMat(-lr, w_gifo_x_corr_);
        w_gifo_r_.AddMat(-lr, w_gifo_r_corr_);
        bias_.AddVec(-lr, bias_corr_, 1.0);
    
        peephole_i_c_.AddVec(-lr, peephole_i_c_corr_, 1.0);
        peephole_f_c_.AddVec(-lr, peephole_f_c_corr_, 1.0);
        peephole_o_c_.AddVec(-lr, peephole_o_c_corr_, 1.0);
    
        w_r_m_.AddMat(-lr, w_r_m_corr_);

//        /* 
//          Here we deal with the famous "vanishing & exploding difficulties" in RNN learning.
//
//          *For gradients vanishing*
//            LSTM architecture introduces linear CEC as the "error bridge" across long time distance
//            solving vanishing problem.
//
//          *For gradients exploding*
//            LSTM is still vulnerable to gradients explosing in BPTT(with large weight & deep time expension).
//            To prevent this, we tried L2 regularization, which didn't work well
//
//          Our approach is a *modified* version of Max Norm Regularization:
//          For each nonlinear neuron, 
//            1. fan-in weights & bias model a seperation hyper-plane: W x + b = 0
//            2. squashing function models a differentiable nonlinear slope around this hyper-plane.
//
//          Conventional max norm regularization scale W to keep its L2 norm bounded,
//          As a modification, we scale down large (W & b) *simultaneously*, this:
//            1. keeps all fan-in weights small, prevents gradients from exploding during backward-pass.
//            2. keeps the location of the hyper-plane unchanged, so we don't wipe out already learned knowledge.
//            3. shrinks the "normal" of the hyper-plane, smooths the nonlinear slope, improves generalization.
//            4. makes the network *well-conditioned* (weights are constrained in a reasonible range).
//
//          We've observed faster convergence and performance gain by doing this.
//        */
//
//        int DEBUG = 0;
//        BaseFloat max_norm = 1.0;   // weights with large L2 norm may cause exploding in deep BPTT expensions
//                                    // TODO: move this config to opts_
//        CuMatrix<BaseFloat> L2_gifo_x(w_gifo_x_);
//        CuMatrix<BaseFloat> L2_gifo_r(w_gifo_r_);
//        L2_gifo_x.MulElements(w_gifo_x_);
//        L2_gifo_r.MulElements(w_gifo_r_);
//
//        CuVector<BaseFloat> L2_norm_gifo(L2_gifo_x.NumRows());
//        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_x, 0.0);
//        L2_norm_gifo.AddColSumMat(1.0, L2_gifo_r, 1.0);
//        L2_norm_gifo.Range(1*ncell_, ncell_).AddVecVec(1.0, peephole_i_c_, peephole_i_c_, 1.0);
//        L2_norm_gifo.Range(2*ncell_, ncell_).AddVecVec(1.0, peephole_f_c_, peephole_f_c_, 1.0);
//        L2_norm_gifo.Range(3*ncell_, ncell_).AddVecVec(1.0, peephole_o_c_, peephole_o_c_, 1.0);
//        L2_norm_gifo.ApplyPow(0.5);
//
//        CuVector<BaseFloat> shrink(L2_norm_gifo);
//        shrink.Scale(1.0/max_norm);
//        shrink.ApplyFloor(1.0);
//        shrink.InvertElements();
//
//        w_gifo_x_.MulRowsVec(shrink);
//        w_gifo_r_.MulRowsVec(shrink);
//        bias_.MulElements(shrink);
//
//        peephole_i_c_.MulElements(shrink.Range(1*ncell_, ncell_));
//        peephole_f_c_.MulElements(shrink.Range(2*ncell_, ncell_));
//        peephole_o_c_.MulElements(shrink.Range(3*ncell_, ncell_));
//
//        if (DEBUG) {
//            if (shrink.Min() < 0.95) {   // we dont want too many trivial logs here
//                std::cerr << "gifo shrinking coefs: " << shrink;
//            }
//        }
//        
    }

private:
    // dims
    int32 ncell_;
    int32 nrecur_;  // recurrent projection layer dim
    int32 nstream_;

    CuMatrix<BaseFloat> prev_nnet_state_;

    // gradient-clipping value,
    BaseFloat clip_gradient_;

    // non-recurrent dropout 
    //BaseFloat dropout_rate_;
    //CuMatrix<BaseFloat> dropout_mask_;

    // feed-forward connections: from x to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_x_;
    CuMatrix<BaseFloat> w_gifo_x_corr_;

    // recurrent projection connections: from r to [g, i, f, o]
    CuMatrix<BaseFloat> w_gifo_r_;
    CuMatrix<BaseFloat> w_gifo_r_corr_;

    // biases of [g, i, f, o]
    CuVector<BaseFloat> bias_;
    CuVector<BaseFloat> bias_corr_;

    // peephole from c to i, f, g 
    // peephole connections are block-internal, so we use vector form
    CuVector<BaseFloat> peephole_i_c_;
    CuVector<BaseFloat> peephole_f_c_;
    CuVector<BaseFloat> peephole_o_c_;

    CuVector<BaseFloat> peephole_i_c_corr_;
    CuVector<BaseFloat> peephole_f_c_corr_;
    CuVector<BaseFloat> peephole_o_c_corr_;

    // projection layer r: from m to r
    CuMatrix<BaseFloat> w_r_m_;
    CuMatrix<BaseFloat> w_r_m_corr_;

    // propagate buffer: output of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> propagate_buf_;

    // back-propagate buffer: diff-input of [g, i, f, o, c, h, m, r]
    CuMatrix<BaseFloat> backpropagate_buf_;

};
} // namespace nnet1
} // namespace kaldi

#endif
