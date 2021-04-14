# QAT_demo
Show how to use QAT in TF1.x to train a network for mnist and convert it into TensorRT engine.

Do it step by step.

- 0.1  `docker pull nvcr.io/nvidia/tensorflow:20.01-tf1-py3`
- 0.2 `docker run --rm -it --gpus all nvcr.io/nvidia/tensorflow:20.01-tf1-py3`
- 0.3 `git clone https://github.com/shiyongming/QAT_demo.git`
- 0.4 `cd QAT_demo/`
- 0.5 `pip install -r requirements.txt`
- 0.6 `cd onnx-graphsurgeon && make install`
- 0.7 `cd ..`


- 1 `python qat_training.py`
- 2 `python export_freezn_graph.py`
- 3 `python fold_constants.py -i saved_results/frozen_graph.pb`
- 4 
```python 
  python3 -m tf2onnx.convert --input saved_results/folded_mnist.pb\
                              --output saved_results/mnist_qat.onnx \
                              --inputs input_0:0 \
                              --outputs softmax_1:0 \
                              --opset 11 
  ```
- 5 `python postprocess_onnx.py --input saved_results/mnist_qat.onnx --output saved_results/mnist_qat_post.onnx`
- 6 `python build_engine.py --onnx saved_results/mnist_qat_post.onnx --engine saved_results/mnist_qat.engine -v`
- 7 `trtexec --loadEngine=saved_results/mnist_qat.engine`
- 8 `python infer_checkAcc.py -e saved_results/mnist_qat.engine -b 1`

The result of Step 8 looks abnormal. It may is casued by the wrong scale of input placeholder. 
