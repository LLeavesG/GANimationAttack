# GANimationAttack

## 说明 


- 该项目是一个能够有效保护用户面部表情不被恶意篡改的面向全民的预防保护平台。产品采用服务端客户端的架构，用户直接面向小程序，降低操作难度。

- 目前对抗Deepfake的主流手段仅有事后检测，即检测图片或视频是否为Deepfake生成的虚假内容，而该项目使用提前保护的策略，提前对要保护的图片内容加入不可见噪点干扰Deepfake生成虚假的模型，切断了虚假内容的生成途径，起到了更好的保护效果。

- 用户将包含人脸的图片由小程序上传到服务器并选择需要保护的表情或者是否进行全局保护的选项。服务端对抗攻击算法将原图片、加入肉眼不可见噪点的图片和选项作为参数传入预训练模型进行对抗样本攻击，经过多轮迭代输出并保存保护后的用户图片、保护后的修改表情效果图片和保护前的修改表情效果对比图。将保护结果返回给用户进行下载，保护前后进行表情修改效果图则仅展示给用户。经过不可见噪点的干扰将会使得Deepfake框架对表情修改后生成明显损坏的图片，达到保护的效果。

- 项目上传部分为后端部分，不包含前端部分

- models中保护预训练模型，针对384*384分辨率，如有需求可自行训练(本项目移除训练相关代码)，请参照原项目`https://github.com/natanielruiz/disrupting-deepfakes`

## Citation
https://github.com/natanielruiz/disrupting-deepfakes
```
@article{ruiz2020disrupting,
    title={Disrupting Deepfakes: Adversarial Attacks Against Conditional Image Translation Networks and Facial Manipulation Systems},
    author={Nataniel Ruiz and Sarah Adel Bargal and Stan Sclaroff},
    year={2020},
    eprint={2003.01279},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
}
