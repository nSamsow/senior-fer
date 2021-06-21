import onnxruntime as rt

sessOptions = rt.SessionOptions()
sessOptions.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL 
chestxrayModel = rt.InferenceSession('model.onnx', sessOptions)

