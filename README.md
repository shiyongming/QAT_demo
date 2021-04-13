# QAT_demo
Show how to use QAT in TF1.x to train a network for mnist and convert it into TensorRT engine.
0.1 pip install -r sampleQAT/requirements.txt
0.2 cd onnx-graphsurgeon && make install
1. python qat_training.py
2. python export_freezn_graph.py
3. python fold_constants.py -i mnist_frozen/frozen_graph.pb
4. python3 -m tf2onnx.convert --input saved_results/folded_mnist.pb --output saved_results/mnist_qat.onnx --inputs input_0:0 --outputs softmax:0 --opset 11
5. python postprocess_onnx.py --input saved_results/mnist_qat.onnx --output saved_results/mnist_qat_post.onnx
6. python build_engine.py --onnx saved_results/mnist_qat_post.onnx --engine saved_results/mnist_qat.engine -v
7. trtexec --loadEngine=saved_results/mnist_qat.engine
