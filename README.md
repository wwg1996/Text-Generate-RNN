# Poetry and novel generate

Using RNN, LSTM automatically generates Chinese poetry texts and Chinese novel texts.

reference:

* [char-rnn-tensorflow](https://github.com/sherjilozair/char-rnn-tensorflow)

* [RNN_poetry_generator](https://github.com/wzyonggege/RNN_poetry_generator)

---

## Poetry
Data：https://github.com/chinese-poetry/chinese-poetry

 * 5.5万首唐诗
 * 26万首宋诗
 * 2.1万首宋词
 * 唐宋两朝近1.4万古诗人
 * 两宋时期1.5K词人

---

## Novel
Data：[贼道三痴](https://baike.baidu.com/item/%E8%B4%BC%E9%81%93%E4%B8%89%E7%97%B4) history novels

 * 《上品寒士》
 * 《雅骚》
 * 《清客》

## USE
### requirements
 * [python 3.5](https://www.python.org/downloads/)
 * [tensorflow](https://www.tensorflow.org/install/) (version >= 1.4)
 * numpy

### sample poetry
```
python generate.py --mode sample --clas poetry
```

```
整首诗生成效果：

《擣练绯》 
噀景抱松草，清风吹颈人。皇军学夸便，和望忆皇恩。砌遍春来老，浮烟菊气长。

《瀑布交山寺肄暕院破田子游诗章二首 二》
何处花林接，伊川实遥早。李朝自二宗，设小都稍醒。乱舞何颓口，垂纶戏耳轻。武者复在宥，见使岂敢尊。

《与溧阳》 
往年南北别，守坐带蝉鸣。上迹报书叶，激然人未归。若逢南岳坑，红换夜嘉仙。

《得韦昭侍御》 
杜鹃筵豁头，忽认巴南天。阴有云斋趣，遥深愁梦梦。髫来住未得，谁屈骏鸟飞。羽檄紫人接，栖景不遑离。遽白李四子，牀余风上船。去名无真州，无复无戡情。柳色自清日，冰毫华素长。所，空木涩。乔楼块盈象，访道想斾浓。白石贫天分，深宫鹤子清。惟应意非时，有事感儒师。
```

```
藏头诗生成效果：

to do
```

### sample novel

```
python generate.py --mode sample --clas novel
```

### train 
 * poetry 
 ```
    python generate.py --mode train --clas poetry
 ```

 * novel
 
 ```
    python generate.py --mode train --clas novel
 ```

### continue train
 * poetry
 ```
    python generate.py --mode con-train --clas poetry
 ```

 * novel
 ```
    python generate.py --mode con-train --clas novel
 ```
