# 申明
该博客所有代码都在colab上运行，关于如何使用colab，可以参考这篇博客。
[colab教程](https://blog.csdn.net/JOHNYXUU/article/details/105870308)
 
 这份baseline来源于vx公众号 **i数据智能**
 比赛期间在这个公众号上学到了很多，在这里表示感谢
 我修改了一些参数，并使用了一些优化方法，提高了分数

最终成绩43名，并不是很高。（第一次参加nlp比赛，还是菜狗）


# 题目描述
[比赛链接](https://www.datafountain.cn/competitions/423)

# 模型分享
## 1 数据分析
数据可以在下面的网址自取
[数据下载地址](https://www.datafountain.cn/competitions/423/datasets)
 ![在这里插入图片描述](https://img-blog.csdnimg.cn/20200501154920370.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0pPSE5ZWFVV,size_16,color_FFFFFF,t_70)
数据内容包括了
['微博id', '微博发布时间', '发布人账号', '微博中文内容', '微博图片', '微博视频', '情感倾向']
### 关于输入
**微博中文内容**是这个模型唯一用到的特征

### 关于图片和视频
其中微博图片和视频是链接，需要自行爬取。
在热心队友的帮助的下我们尝试了如下方法：

**1 使用ocr方法提取图片上的文字**

**2使用image caption方法把图片转为文字描述**

可惜的是队员们家里的gpu效果很差，最好的也只有1060，所以花费了几天的时间才把所有的数据跑完。
至于为什么不在colab做这个图片的处理呢，是因为把图片上传到colab实在是太慢了。
而且我们做文字描述的时候需要修改keras的源文件，所以决定在本地跑的。
并且最终融合到我的模型里也没有产生好的效果，所以后来也没用上。

关于视频特征，由于缺失值太多，加上本来gpu资源也落后，也没有使用。

### 关于输出

情感倾向就是我们要预测的Y值了，分为三类-1 消极 0 中性 1积极

所以就是一个三分类的大数据算法类题目。

## 2 模型选取
选用了大热的bert模型，语料包也用的官方版本

[bert github地址](https://github.com/huggingface/transformers)

我也使用过哈工大的RoBERTa-wwm-ext等模型
但效果都没有bert好
可能是我的显卡显存太小了只有16g
难以放下过大的batch_size 导致了效果不好
[BERT-wwm github地址](https://github.com/ymcui/Chinese-BERT-wwm)

## 3 输入数据处理
```python
def _convert_to_transformer_inputs(instance, tokenizer, max_sequence_length):
    """Converts tokenized input to ids, masks and segments for transformer (including bert)"""
    
    def return_id(str1, truncation_strategy, length):

        inputs = tokenizer.encode_plus(str1,
            add_special_tokens=True,
            max_length=length,
            truncation_strategy=truncation_strategy)
        
        input_ids =  inputs["input_ids"]
        input_masks = [1] * len(input_ids)
        input_segments = inputs["token_type_ids"]
        padding_length = length - len(input_ids)
        padding_id = tokenizer.pad_token_id
        input_ids = input_ids + ([padding_id] * padding_length)
        input_masks = input_masks + ([0] * padding_length)
        input_segments = input_segments + ([0] * padding_length)
        
        return [input_ids, input_masks, input_segments]
    
    input_ids, input_masks, input_segments = return_id(
        instance, 'longest_first', max_sequence_length)
    
    return [input_ids, input_masks, input_segments]

def compute_input_arrays(df, columns, tokenizer, max_sequence_length):
    input_ids, input_masks, input_segments = [], [], []
    for instance in tqdm(df[columns]):
        
        ids, masks, segments = \
        _convert_to_transformer_inputs(str(instance), tokenizer, max_sequence_length)
        
        input_ids.append(ids)
        input_masks.append(masks)
        input_segments.append(segments)

    return [np.asarray(input_ids, dtype=np.int32), 
            np.asarray(input_masks, dtype=np.int32), 
            np.asarray(input_segments, dtype=np.int32)
           ]
```
这是处理中文语句的函数

```python
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
inputs = compute_input_arrays(df_train, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
test_inputs = compute_input_arrays(df_test, input_categories, tokenizer, MAX_SEQUENCE_LENGTH)
```
使用了官方的分词器 **BertTokenizer**
这一步做了如下的工作

> 1 将中文文本转换为词向量，说白了就是把中文用数字代替
> 2 将此向量转换为bert模型的三个输入 ids masks segments

接下来就可以把生成的数据用作输入数据了

## 4 建立模型

```python
def create_model():
    input_id = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)    
    input_mask = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32)    
    input_atn = tf.keras.layers.Input((MAX_SEQUENCE_LENGTH,), dtype=tf.int32) 
    config = BertConfig.from_pretrained('bert-base-chinese')
    bert_model = TFBertModel.from_pretrained('bert-base-chinese',config=config)
    embedding = bert_model(input_id, attention_mask=input_mask, token_type_ids=input_atn)[0]   
    x = tf.keras.layers.GlobalAveragePooling1D()(embedding)    
    x = tf.keras.layers.Dropout(0.2)(x)
    x1 = tf.keras.layers.Dense(3, activation='softmax',name='class_out')(x)
    model = tf.keras.models.Model(inputs=[input_id, input_mask, input_atn], outputs=x1)
    return model
```
config 是用来下载bert-chinese的模型参数的，本地下的非常慢，但colab网速给力。
中间的dropout是为了防止模型过拟合，多次调试发现0.2效果最佳
softmax在分类问题中效果较好，所以选择了它作为激活层

## 5 得出初步结果

```python
gkf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))
valid_preds = []
test_preds = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
        train_inputs = [inputs[i][train_idx] for i in range(len(inputs))]
        train_outputs = to_categorical(outputs[train_idx])
        valid_inputs = [inputs[i][valid_idx] for i in range(len(inputs))]
        valid_outputs = to_categorical(outputs[valid_idx])
        K.clear_session()
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        model = create_model()
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['acc', 'mae'])
        model.summary()
        model.fit(train_inputs, train_outputs, validation_data= [valid_inputs, valid_outputs], epochs=1,batch_size=64)
        valid_preds.append(model.predict(valid_inputs))
        test_preds.append(model.predict(test_inputs))
```
优化器：Adam 优化效果很好的优化器，貌似大家都在用它
损失函数：categorical_crossentropy 也是在多分类问题中效果显著，所以选用了他
分了5个folds，每次都预测一次test，最后取平均值得到predict
epoch取了1，因为多次会过拟合，效果很差

最后根据三个结果的分数，哪个高选哪个就可以得到一个结果
**到这一步线上已经可以超过0.73了**

## 6 f1_score 优化
这一步同样来自公众号**i数据智能**的文章，再次表示感谢

```python
class OptimizedRounder(object):
    def __init__(self):
        self.coef_ = [1.,1.,1.]
    def _kappa_loss(self, coef, X, y):
        X_p = np.copy(X)
        X_p = np.argmax(X_p*coef,axis=1)
        y_t = np.argmax(y,axis=1)
        ll = f1_score(y_t, X_p,average='macro')
        print('f1 score: ',ll)
        return -ll
    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        if type(self.coef_) is list:
          initial_coef = self.coef_
        else:
          initial_coef = self.coef_['x']
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef,method='Nelder-Mead')
    def predict(self, X, coef):
        X_p = np.copy(X)
        X_p = np.argmax(X_p*coef,axis=1)
        return X_p
    def coefficients(self):
        return self.coef_['x']
```

```python
gkf = StratifiedKFold(n_splits=5,shuffle=True,random_state=0).split(X=df_train[input_categories].fillna('-1'), y=df_train[output_categories].fillna('-1'))
for fold, (train_idx, valid_idx) in enumerate(gkf):
  print('flod: ',fold)
  valid_outputs = to_categorical(outputs[valid_idx])
  opr.fit(X=valid_preds[fold],y=valid_outputs)
```
我们可以使用scipy库中的optimize函数
拟合结果，使得我们的f1_score提高
最后分数提高了6个千分点
来到了0.7389
也是我的最高分数

## PS：
bert模型很吃batchsize，需要较大的batch_size才能得到较好结果
