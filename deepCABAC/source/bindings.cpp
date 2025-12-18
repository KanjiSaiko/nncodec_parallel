#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <Lib/CommonLib/TypeDef.h>
#include <Lib/CommonLib/Quant.h>
#include <Lib/EncLib/CABACEncoder.h>
#include <Lib/DecLib/CABACDecoder.h>
#include <iostream>
#include <math.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;
py::list quantize_all_blocks_parallel_pthreads(py::list py_block_info_list);

class Encoder
{
public:
  Encoder() { m_CABACEncoder.startCabacEncoding( &m_Bytestream ); }
  ~Encoder() {}
  void                  initCtxModels(uint32_t cabac_unary_length_minus1, uint8_t param_opt_flag) { m_CABACEncoder.initCtxMdls(cabac_unary_length_minus1+1, param_opt_flag); }
  void                  iae_v( uint8_t v, int32_t value )            { m_CABACEncoder.iae_v( v, value ); }
  void                  uae_v( uint8_t v, uint32_t value )           { m_CABACEncoder.uae_v( v, value ); }
  uint32_t              encodeLayer( py::array_t<int32_t, py::array::c_style> qindex, uint8_t dq_flag, int32_t scan_order  );
  int32_t               quantLayer( py::array_t<float32_t, py::array::c_style> Weights, py::array_t<int32_t, py::array::c_style> qIndex, uint8_t dq_flag, int32_t qpDensity, int32_t qp,float32_t lambdaScale, uint32_t maxNumNoRem, int32_t scan_order );
  py::array_t<uint8_t>  finish();
private:
  std::vector<uint8_t>  m_Bytestream;
  CABACEncoder          m_CABACEncoder;
};

int32_t Encoder::quantLayer(py::array_t<float32_t, py::array::c_style> Weights, py::array_t<int32_t, py::array::c_style> qIndex, uint8_t dq_flag, int32_t qpDensity, int32_t qp, float32_t lambdaScale, uint32_t maxNumNoRem, int32_t scan_order )
{
  py::buffer_info bi_Weights = Weights.request();
  py::buffer_info bi_qIndex = qIndex.request();
  float32_t* pWeights          = (float32_t*) bi_Weights.ptr;
  int32_t* pQIndex = (int32_t*) bi_qIndex.ptr;

  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for (size_t idx = 0; idx < (size_t)bi_Weights.ndim; idx++)
  {
    numWeights *= bi_Weights.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_Weights.shape[idx];
  }
  if( layerWidth == 1 || numWeights == layerWidth )
      scan_order = 0;
      
  int32_t k = 1 << qpDensity;
  int32_t mul = k + (qp & (k-1));
  int32_t shift = qp >> qpDensity;
  float32_t qStepSize = mul * pow(2.0, shift - qpDensity);

  int32_t success = quantize(pWeights, pQIndex, qStepSize, layerWidth, numWeights, DIST_MSE, lambdaScale, dq_flag, maxNumNoRem, scan_order);

  if( !success )
  {
    float32_t maxAbs = 0.0;

    for(uint32_t i = 0; i < numWeights; i++)
    {
      if( abs( pWeights[i] ) > maxAbs )
      {
        maxAbs = abs(pWeights[i]);
      }
    }

    double minStepsize = (double)(maxAbs) / ((double)((1u << 31) - 3));

    float32_t baseQP = floor(log2(minStepsize)) * k;
    float32_t newQp = baseQP + ((minStepsize * k) / pow(2.0, (baseQP / k)) - k);
    qp = (int32_t)(ceil(newQp));

    mul = k + (qp & (k - 1));
    shift = qp >> qpDensity;
    qStepSize = mul * pow(2.0, shift - qpDensity);

    success = quantize(pWeights, pQIndex, qStepSize, layerWidth, numWeights, DIST_MSE, lambdaScale, dq_flag, maxNumNoRem, scan_order);
    CHECK( !success, "Prevention of integer-overflow failed!");
  }
  return qp;
}

uint32_t Encoder::encodeLayer( py::array_t<int32_t, py::array::c_style> qindex, uint8_t dq_flag, int32_t scan_order )
{
  py::buffer_info bi_qindex = qindex.request();
  int32_t* pQindex          = (int32_t*) bi_qindex.ptr;

  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for( size_t idx = 0; idx < (size_t)bi_qindex.ndim; idx++ )
  {
    numWeights *= bi_qindex.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_qindex.shape[idx];
  }
  if( layerWidth == 1 || numWeights == layerWidth )
      scan_order = 0;

  return m_CABACEncoder.encodeWeights(pQindex, layerWidth, numWeights, dq_flag, scan_order);
}

py::array_t<uint8_t> Encoder::finish()
{
  m_CABACEncoder.terminateCabacEncoding();

  auto Result = py::array_t<uint8_t, py::array::c_style>(m_Bytestream.size());
  py::buffer_info bi_Result = Result.request();
  uint8_t* pResult = (uint8_t*) bi_Result.ptr;

  for( size_t idx = 0; idx < m_Bytestream.size(); idx ++ )
  {
    pResult[idx] = m_Bytestream.at(idx);
  }
  return Result;
}

class Decoder
{
public:
  Decoder() {}
  ~Decoder() {}

  void     setStream    ( py::array_t<uint8_t, py::array::c_style> Bytestream );
  void     initCtxModels( uint32_t cabac_unary_length_minus1 ) { m_CABACDecoder.initCtxMdls( cabac_unary_length_minus1+1 ); }
  int32_t  iae_v        (uint8_t v) { return m_CABACDecoder.iae_v(v); }
  uint32_t uae_v        ( uint8_t v )                   { return m_CABACDecoder.uae_v( v ); }

  py::array_t<uint64_t> decodeLayerAndCreateEPs(py::array_t<int32_t, py::array::c_style> Weights, uint8_t dq_flag, int32_t scan_order); //Return value -> Array? Ptr?
  void     setEntryPoints( py::array_t<uint64_t, py::array::c_style> entryPoints);
  void     decodeLayer  ( py::array_t<int32_t, py::array::c_style> Weights, uint8_t dq_flag, int32_t scan_order );
  void     dequantLayer ( py::array_t<float32_t, py::array::c_style> Weights, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order);
  uint32_t finish       ();

private:
  CABACDecoder  m_CABACDecoder;
};

void Decoder::setStream( py::array_t<uint8_t, py::array::c_style> Bytestream )
{
  py::buffer_info bi_Bytestream = Bytestream.request();
  uint8_t* pBytestream          = (uint8_t*) bi_Bytestream.ptr;
  m_CABACDecoder.startCabacDecoding( pBytestream );
}

py::array_t<uint64_t> Decoder::decodeLayerAndCreateEPs(py::array_t<int32_t, py::array::c_style> Weights, uint8_t dq_flag, int32_t scan_order)
{
  std::vector<uint64_t> entryPoints; 
  py::buffer_info bi_Weights = Weights.request();

  int32_t *pWeights = (int32_t *)bi_Weights.ptr;
  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for (size_t idx = 0; idx < (size_t)bi_Weights.ndim; idx++)
  {
    numWeights *= bi_Weights.shape[idx];
    if (idx == 0)
    {
      continue;
    }
    layerWidth *= bi_Weights.shape[idx];
  }
  if (layerWidth == 1 || numWeights == layerWidth)
    scan_order = 0;

  m_CABACDecoder.decodeWeightsAndCreateEPs(pWeights, layerWidth, numWeights, dq_flag, scan_order, entryPoints);

  auto Result = py::array_t<uint64_t, py::array::c_style>(entryPoints.size());
  py::buffer_info bi_Result = Result.request();
  uint64_t *pResult = (uint64_t *)bi_Result.ptr;

  for (size_t idx = 0; idx < entryPoints.size(); idx++)
  {
    pResult[idx] = entryPoints.at(idx);
  }

  return Result;
}

void Decoder::setEntryPoints(py::array_t<uint64_t, py::array::c_style> entryPoints)
{
  py::buffer_info bi_EntryPoints = entryPoints.request();

  uint64_t *pEntryPoints = (uint64_t *)bi_EntryPoints.ptr;
  uint64_t numEntryPoints = 1;

  for (size_t idx = 0; idx < (size_t)bi_EntryPoints.ndim; idx++)
  {
    numEntryPoints *= bi_EntryPoints.shape[idx];
  }

  m_CABACDecoder.setEntryPoints(pEntryPoints, numEntryPoints);
}

void Decoder::decodeLayer( py::array_t<int32_t, py::array::c_style> Weights , uint8_t dq_flag, int32_t scan_order )    
{
  py::buffer_info bi_Weights = Weights.request();

  int32_t* pWeights   = (int32_t*) bi_Weights.ptr;
  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for (size_t idx = 0; idx < (size_t)bi_Weights.ndim; idx++)
  {
    numWeights *= bi_Weights.shape[idx];
    if( idx == 0 ) { continue; }
    layerWidth *= bi_Weights.shape[idx];
  }
  if( layerWidth == 1 || numWeights == layerWidth )
      scan_order = 0;
  m_CABACDecoder.decodeWeights(pWeights, layerWidth, numWeights, dq_flag, scan_order);
}


void Decoder::dequantLayer(py::array_t<float32_t, py::array::c_style> Weights, py::array_t<int32_t, py::array::c_style> qIndex, int32_t qpDensity, int32_t qp, int32_t scan_order)
{
  py::buffer_info bi_Weights = Weights.request();
  py::buffer_info bi_qIndex = qIndex.request();

  float32_t *pWeights = (float32_t *)bi_Weights.ptr;
  int32_t *pQIndex = (int32_t *)bi_qIndex.ptr;
  uint32_t layerWidth = 1;
  uint32_t numWeights = 1;
  for (size_t idx = 0; idx < (size_t)bi_Weights.ndim; idx++)
  {
    numWeights *= bi_Weights.shape[idx];
    if (idx == 0)
    {
      continue;
    }
    layerWidth *= bi_Weights.shape[idx];
  }
  if( layerWidth == 1 || numWeights == layerWidth )
      scan_order = 0;

  int32_t k = 1 << qpDensity;
  int32_t mul = k + (qp & (k-1));
  int32_t shift = qp >> qpDensity;
  float32_t qStepSize = mul * pow(2.0, shift - qpDensity);
  deQuantize(pWeights, pQIndex, qStepSize, numWeights, layerWidth, scan_order);
}


uint32_t Decoder::finish()
{
  uint32_t bytesRead = m_CABACDecoder.terminateCabacDecoding();
  return bytesRead;
}


PYBIND11_MODULE(deepCABAC, m) 
{
    py::class_<Encoder>(m, "Encoder")
        .def( py::init<>())
        .def( "iae_v",         &Encoder::iae_v         )
        .def( "uae_v",         &Encoder::uae_v         )
        .def( "initCtxModels", &Encoder::initCtxModels )
        .def( "quantLayer",    &Encoder::quantLayer    )
        .def( "encodeLayer",   &Encoder::encodeLayer   )
        .def( "finish",        &Encoder::finish        );

    py::class_<Decoder>(m, "Decoder")
        .def( py::init<>())
        .def( "setStream",     &Decoder::setStream, py::keep_alive<1, 2>() )
        .def( "initCtxModels", &Decoder::initCtxModels )
        .def( "iae_v",         &Decoder::iae_v         )
        .def( "uae_v",         &Decoder::uae_v         )
        .def( "decodeLayer",   &Decoder::decodeLayer   )
        .def( "decodeLayerAndCreateEPs",   &Decoder::decodeLayerAndCreateEPs   )
        .def( "setEntryPoints",&Decoder::setEntryPoints)
        .def( "dequantLayer",  &Decoder::dequantLayer  )
        .def( "finish",        &Decoder::finish        );

    m.def("quantize_all_blocks_parallel", 
          &quantize_all_blocks_parallel_pthreads, 
          "Parallel quantization of multiple blocks using pthreads",
          py::arg("block_info_list"));
}
