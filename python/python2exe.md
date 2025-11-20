
```
pip install pyinstaller
```


```
pyinstaller --onefile --add-data "path/to/your/models;models" main.py
```
* --onefile 打包成单个 exe
* --add-data 用来把模型、权重等文件打包到 exe 里，格式 源路径;打包内路径
  *  Windows 下用分号 ;，Linux/Mac 用冒号 :

```
pyinstaller --onefile --name ocr_app --add-data "weights;weights" --add-binary "E:/anaconda3/envs/py39_onnx/Lib/site-packages/shapely/.libs/*;shapely/.libs" --add-binary "E:/anaconda3/envs/py39_onnx/Lib/site-packages/onnxruntime/onnxruntime.dll;onnxruntime" main.py


pyinstaller --onefile --name ocr_app --add-data "weights;weights" --add-binary "E:/anaconda3/envs/py39_onnx/Lib/site-packages/shapely/.libs;shapely/.libs" --add-binary "E:/anaconda3/envs/py39_onnx/Lib/site-packages/onnxruntime/onnxruntime.dll;onnxruntime" main.py


pyinstaller --onefile --name ocr_app --add-data "weights;weights" main.py

python main.py --input D:\github\PaddleOCR-cpp\paddleOCR_win\images\test.bmp

python main.py --input D:\gitlab\ocr_cpp\test\imgs\test4.png
```


