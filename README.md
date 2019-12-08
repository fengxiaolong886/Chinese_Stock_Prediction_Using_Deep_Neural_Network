# Abstract
In this research, we study the problem of Chinese stock market forecasting using traditional Neural Network methods, including Deep Feedforward Network, Convolution Neural Network(CNN), Recurrent Neural Network(RNN), Long Short-Term Memory(LSTM) and we have also integrate with the Bi-direction technology. The purpose of this research is not to find a state-of-art algorithm to make the prediction more precise. We focus on the primary method and explore the relationship between architecture and hyper-parameter(including learning rate, variety of activation ,and variety of the output target designed. Due to the advanced of the Deep Learning method in decades, there is much active research in the Time-Serial Analysis domain, but we rarely not sure whether they are work in reality. The real market is a very sophisticated ecosystem, and many accidents may occur, influent the market price. Even there is an efficient market hypothesis from Eugene Fama, but we should not so sure we can control the market by this theoretical analysis. Although we implement the underlying deep neural network architecture, the experiment has a reassuring result with $31.9\%$ accuracy in the eight classes classification task and $68.6\%$ accuracy in the binary classification task.

# Introduction
## Neural Network and Deep Learning
Deep learning is a member of machine learning. It is a re-branded name for neural networks—a family of learning techniques that were historically motivated by the way computation works in the brain, and which can be described as learning of parameterized differentiable mathematical functions. The name deep-learning arises from the fact that many layers of these differentiable functions are often attached together. While all of the machine learning can be defined as learning to make forecasts based on past observations, deep learning approaches work by learning to not only predict but also to exactly represent the data, such that it is suitable for the forecast. Given a broad set of desired input-output mapping, deep learning approaches work by feeding the data into a network that produces progressive transformations of the input data until a final transformation predicts the output. The transformations produced by the network are learned from the given input-output mappings, such that each transformation makes it easier to relate the data to the desired label.

While the human designer is in charge of designing the network architecture and training regime, providing the network with a proper set of input-output examples, and encoding the input data in a proper way, a lot of the heavy-lifting of learning the exact representation is performed automatically by the network, supported by the network's architecture.

## Recent Advances in Deep Learning
Deep Learning is one of the charming fields in recent years(Goodfellow u.a., 2016). The breakthrough of Deep Learning since the success of AlexNet t (Creswell u.a., 2017) in 2012. Deep Learning is used in the domain of digital image processing to solve severe problems(e.g., image colorization, classification, segmentation, and detection). In that domain, methods such as Convolutional Neural Networks(CNN) mostly improve prediction performance using big data and unlimited computing resources and have pushed the boundaries of what was possible.

Deep Learning has pushed the boundaries of what was possible in the field of Machine Learning and Artificial Intelligence. Now and then, new and new deep learning techniques are being yielded, exceeding state-of-the-art deep learning techniques. Deep Learning is evolving at a considerable speed, its kind of hard to keep track of the regular advances. In our experiment, we are going to briefly review the basic idea in Deep Learning used in the Chinese stock market.

However, that is not so to say that the traditional Machine Learning techniques  (Cao u. Tay, 2003),(Kercheval u. Yuan, 2015),(Zhai u.a., 2007),(Armano u.a., 2005) which had been undergoing continuous improvement in years before the rise of Deep Learning have become obsolete. But in this paper, we will not review the advantages and disadvantages of each approach. We only focus on the method of Deep Learning applied to the time series problem.

## Financial Market Prediction and Time Series Analysis
Modeling and Forecasting of the financial market have been a trendy topic for scholars and researchers. For the efficient markets hypothesis, proposed by Eugene Fama, who is one of the winners of Nobel economics prize in 2013. Although we are not sure about Fama's theory, we should know that Lars peter Hansen shared the same prize at the same time. That mean peoples are confused the question about whether the market could be predicted. Nevertheless, the application for the time series problem is not only used in the market, it also could be used in weather forecasting, statistics, signal processing, pattern recognition, earthquake prediction, electroencephalography, control engineering, astronomy, communications engineering, and primarily in any field applied science and engineering which requires temporal measurements.

Time series analysis comprises methods for analyzing time-series data in order to extract meaningful statistics and other characteristics of the data. Time series forecasting is the application of a model to predict future values based on previously observed values.

## Dataset Description
Modern Deep Network method is supported by big data and high-speed data process capability. For the aspect of the data, stock market data is the natural source of the research target. Notably, we dived into the stock market data in China. To find out how the stock market works in China, we should be at first mentioned where the stocks are traded. Thus, we can trade Chinese stocks on the most popular stock exchanges of China: Shanghai Stock Exchange (SSE) and the Shenzhen Stock Exchange (SZSE).

In our experiment, we collect all the daily trade data in the past three decades. The data starts with the first stock released and ends in December 2019. We do not consider the subset of all stocks, but all the stocks data are considered in our experiment. Considering all stocks helps us analyze the internal laws of the Chinese stock market. Because of the difficulty of data processing, we do not use the minute level and hour level data but use the transaction day level data. Although the time scale of the amount of data we use is large, the amount of data at this scale still exceeds 10 million records.

In order to have an intuitive understanding of China’s stock market, we draw the trend chart of the shanghai composite index in Fig1.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-e40917665d1f7a54.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# Related Work
Financial time-series prediction is vital for developing excellent trading strategies in the financial market  (Wang, 2014). In past decades, it has attracted much attention from researchers of many fields, especially the Machine Learning community  (Lee u. Ready, 1991). These researches mainly focus on a specific market, e.g., the stock market (Kim, 2003)(Li u.a., 2016), the foreign exchange market (Lin u.a., 2017),(Atsalakis u. Valavanis, 2009), and the futures market t (Cheung u.a., 2019) ,(Kimu.a., 2017). Unsurprisingly, this is a holy grail challenge in the finance field due to their irregular and noisy environment.

From the perspective of the learning target, existing researches can be divided into the regression approaches and classification approaches. For the regression approaches, they treat this task as a regression problem  (Zirilli, 1996),(Bollen u.a., 2011), aiming to predict the future value of financial time-series. While the classification-oriented approaches treat this as a classification problem (Schumaker u. Chen, 2009),(Hsieh u.a., 2011) focusing on financial time-series classification (FTC).

In most cases, the classification approaches achieve higher profits than the regression ones  (Huangu.a., 2008). Accordingly, the effectiveness of various approaches in FTC has been widely explored (Li u.a., 2016),(Leung u.a., 2000).
There were also many overview papers on Deep Learning (DL) in the past years. They described DL methods and approaches in significant ways as well as their applications and directions for future research.

In  (Young u.a., 2017), the researchers talked about DL models and architectures, mainly used in Natural Language Processing (NLP). They showed DL applications in various NLP fields, compared DL models, and discussed possible future trends. Furthermore, in  (Goodfellow u.a., 2016), the researchers discussed deep networks and generative models in detail. Starting from Machine Learning (ML) basics, pros, and cons for deep architectures, they concluded recent DL researches and applications thoroughly.

# Advances In Deep Learning
In this section, we will discuss the leading recent Deep Learning (DL) approaches derived from Machine Learning and brief evolution of Artificial Neural Networks (ANN), which is the most common form used for deep learning.

## Architecture
### Multi Layer Perceptron
As we abandon the brain metaphor and describe networks exclusively in terms of vector-matrix operations. The simplest neural network is called a perceptron. It is simply a linear model is:
$$
\begin{align}
	NN_{Perceptron}(x)=xW + b \\  s.t. \ \ \ \ x \in R^{d_{in}},W \in R^{d_{in}d_{out}},b \in R^{d_{out}}
\end{align}
$$
Where W is the weight matrix, and b is a bias term. In order to go beyond linear functions, we introduce a nonlinear hidden layer, resulting in the Multi Layer Perceptron. A feed-forward neural network with two hidden-layer has the form as :
$$
\begin{align}
	MLP_2(x)=(g^2(g^1(xW^1+b^2)W^2+b^2))W^3\\s.t. \ \ \ x \in R^{d_{in}},W^1 \in R^{d_{in} \times d_{1}},b^1 \in R^{d_1},W^2 \in R^{d_{1} \times d_{2}},b^2 \in R^{d_{2}}
\end{align}
$$
Where g is the nonlinear function, which could be relu, tanh, sigmoid, and so on.
### The CNN architecture
The convolution-and-pooling (also called convolutional neural networks, or CNNs) architecture, which is tailored to this modeling problem. A convolutional neural network is designed to identify indicative local predictors in a large structure, and to combine them to produce a fixed-size vector representation of the structure, capturing the local aspects that are most informative for the prediction task at hand.

The CNN is, in essence, a feature-extracting architecture. It does not constitute a standalone, useful network on its own, but rather is meant to be integrated into a more extensive network, and to be trained to work in tandem with it in order to produce an end result. The CNN layer's responsibility is to extract meaningful sub-structures that are useful for the overall prediction task at hand.

Convolution-and-pooling architectures evolved in the neural network's vision community, where they showed great success as object detectors—recognizing an object from a predefined category (“cat,” “bicycles”) regardless of its position in the image  (Wang u. Choi, 2013) When applied to images, the architecture is using 2D (grid) convolutions. When applied to the time series problem, we have mainly concerned with 1D (sequence) convolutions. Because of their origins in the computer-vision community, a lot of the terminology around convolutional neural networks is borrowed from computer vision and signal processing, including terms such as filter, channel, and receptive-field.
We can have an intuitive understanding of CNN from Fig 2.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-c76fa5ca8e023af3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


### The RNN architecture
As describes RNNs as an abstraction: an interface for translating a sequence of inputs into a fixed sized output that can then be plugged as components in larger networks. Various architectures that use RNNs as a component are discussed.
We use $x_{i:j}$ to denote the sequence of vectors $x_i,\cdot\cdot\cdot,x_j$ . On a high-level, the RNN is a function that takes as input an arbitrary length ordered sequence of $n$ $d_{in}$ dimensional vectors $x_{1:n}=x_1,x_2\cdot\cdot\cdot\cdot x_n$,($x_i \in R^{d_{in}}$) and returns as output a single $d_{out}$ dimensional vector $y_n \in R^{d_{oiut}}$:
$$
\begin{align}
	y_n=RNN(x_{1:n}).\\  s.t.  \ \ \ \ \ x \in R^{d_{in}} ,y_n \in R^{d_{out}}
\end{align}
$$
This implicitly defines an output vector$y_i$ for each prefix $x_{1:i}$ of the sequence x $x_{1:n}$. We denote by $RNN^*$ the function returning this sequence:
$$
\begin{align}
	& y_{1:n}=RNN^*(x_{1:n})\\  & y_i=RNN(x_{1:i})\\ &s.t.  x_i\in R^{d_{in}},y_n \in R^{d_{out}}
\end{align}
$$
The output vector $y_n$  is then used for further prediction. For example, a model for predicting the conditional probability of an event e given the sequence $x_{1:n}$ can be defined as the equation below.
$$
\begin{align}
	p(e=j|x_{1:n})={softmax(RNN(x_{1:n})\cdot W+b)}_{[j]}
\end{align}
$$
The $j_{th}$ element in the output vector resulting from the softmax operation over a linear transformation of the RNN encoding.
$$
\begin{align}
	y_n=RNN(x_{1:n})
\end{align}
$$
We can denote the RNN as the form of recursive as Fig3.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-5ea9853484595ece.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**Bidirectional RNNs (BIRNN)**
A useful elaboration of an RNN is a bidirectional-RNN (also commonly referred to as biRNN) [20 ]. Consider an input sequence $x_{1:n}$. The biRNN works by maintaining two separate states, $s^f_i$, and $s^b_j$ for each input position $i$ . The forward state $s^f_i$ is based on $x_1,x_2\cdot\cdot\cdot\cdot x_i$ , while the backward state $s^b_j$ is based on $x_n,x_{n-1}\cdot\cdot\cdot\cdot x_i$. The forward and backward states are generated by two different RNNs. The first $RNN(R^f,O^f)$ is fed the input sequence $x_{1:n}$ as is, while the second $RNN(R^b,O^b)$ is fed the input sequence in reverse. The state representation $s_i$ is then composed of both the forward and backward states. The output at position $i$ is based on the concatenation of the two output vectors $y_i=[y^f_i:y^b_i]=[O^f(s^f_i):O^b(s^b_i)]$, taking into account both the past and the future. In other words, $y_i$ , the biRNN encoding of the $i_{th}$ word in a sequence is the concatenation of two RNNs, one reading the sequence from the beginning, and the other reading it from the end.
We define $biRNN(x_{1:n},i)$ to  be the output vector corresponding to the $i_{th}$ sequence position:
$$
\begin{align}
	biRNN(x_{1:n},i)=y_i=[RNN^f(x_{1:i}):RNN^b(x_{x_{n:i}})]
\end{align}
$$
The vector $y_i$ can then be used directly for prediction or fed as part of the input to a more complex network. While the two RNNs are run independently of each other, the error gradients at position $i$ will flow both forward and backward through the two RNNs. Feeding the vector $y_i$  through an MLP prior to prediction will further mix the forward and backward signals. Visual representation of the biRNN architecture is given in Fig 4.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-d575dc8dadcad022.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


**Multi-Layer(Stacked) RNNs**
RNNs can be stacked in layers, forming a grid  (Hihi u. Bengio, 1995). Consider $k$ RNNs, $RNN_1\cdot\cdot\cdot\cdot RNN_k$, where the $j_{th}$ RNN has states $s^j_{1:n}$ and outputs $y^j_{1:n}$. The input for the first RNN is $x_{1:n}$, while the input of the $j_{th}$ $RNN ( j \geq 2 $) are the outputs of the RNN below it $y^{j-1}_{1:n}$, The output of the entire formation is the output of the last RNN, $y^k_{1:n}$. Such layered architectures are often called deep RNNs. A visual representation of a three-layer RNN is given in Fig5.
 ![image.png](https://upload-images.jianshu.io/upload_images/15463866-710061e7fe1c95fa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

**Simple RNN**
The simplest RNN formulation that is sensitive to the ordering of elements in the sequence is known as an Elman Network or Simple-RNN (S-RNN). The S-RNN was proposed by (Elman, 1990). The S-RNN takes the following form:
$$
\begin{align}
	&s_{i}=R_{SRNN}(x_i,s_{i-1})=g(s_{i-1}W^s+x_iW^x+b) \\&y_i=O(s_{i})\\ &s.t. \ \ \  s_i,y_i\in R^{d_{s}},x_i \in R^{d_{x}},W^x \in R^{d_x \times d_{s}},W^s \in R^{d_s \times d_{s}},b \in R^{d_s}
\end{align}
$$
That is, the state $s_{i-1}$ and the input $x_i$ are each linearly transformed, the results are added(together with a bias term) and then passed through a nonlinear activation function $g$ (commonly tanh or ReLU). The output at position $i$ is the same as the hidden state in that position.
**Gated Architectures**
The S-RNN is hard to train effectively because of the vanishing gradients problem  (Pascanu u.a., 2012). Error signals (gradients ) in later steps in the sequence diminish quickly in the back-propagation process and do not reach earlier input signals, making it hard for the S-RNN to capture long-range dependencies. Gating-based architectures, such as the LSTM  (Hochreiter u.Schmidhuber, 1997) and the GRU  (Cho u.a., 2014a) are designed to solve this deficiency.
**LSTM**
The Long Short-Term Memory (LSTM) architecture (Hochreiteru. Schmidhuber,1997) was designed to solve the vanishing gradients problem and is the first to introduce the gating mechanism. The LSTM architecture explicitly splits the state vector $s_i$ into two halves, where one half is treated as “memory cells” and the other is working memory. The memory cells are designed to preserve the memory, and also the error gradients, across time, and are controlled through differentiable gating components —smooth mathematical functions that simulate logical gates. At each input state, a gate is used to decide how much of the new input should be written to the memory cell, and how much of the current content of the memory cell should be forgotten. Mathematically, the LSTM architecture is defined as:
$$
\begin{align}
	s_j=R_{LSTM}(S_{j-1},x_j&)=[c_j:h_j] \\&c_j=f \odot c_{j-1} +i \odot z \\ &h_j=o \odot tanh(c_j)\\&i=\delta (x_j W^{xi}+h_{j-1}W^{hi})\\&f=\delta (x_j W^{xf}+h_{j-1}W^{hf})\\&o=\delta (x_j W^{xo}+h_{j-1}W^{ho})\\&z=tanh(x_j W^{xz}+h_{j-1}W^{hz})\\y_j=&O_{LSTM}(s_j)=h_j\\  s.t. \ \ s_j\in R^{2 \cdot{d_{h}}},x_i \in R^{d_{x}},c_j,h_j,i,f,o,&z \in R^{d_h},W^s \in R^{d_s \times d_{s}},W^{h_0} \in R^{d_h \times d_{h}}
\end{align}
$$
The state at time $j$ is composed of two vectors, $c_j$, and $h_j$, where $c_j$ is the memory component, and $h_j$ is the hidden state component. There are three gates, $i$ , $ f $ , and $o$ , controlling for input, forget, and output. The gate values are computed based on linear combinations of the current input $x_j $and the previous state $h_{j-1}$ , passed through a sigmoid activation function. An update candidate $z$ is computed as a linear combination of $x_j $ and $h_{j-1}$ , passed through a tanha ctivation function. The memory $c_j$ is then updated: the forget gate controls how much of the previous memory to keep $f\odot c_{j-1}$, and the input gate controls how much of the proposed update to keep $i \odot z$. Finally, the value of $ h_j$ (which is also the output $y_j$ ) is determined based on the content of the memory $c_j$, passed through a tanh nonlinearity, and controlled by the output gate. The gating mechanisms allow for gradients related to the memory part $c_j$ to stay high across very long time ranges.
LSTMs are currently the most successful type of RNN architecture, and they are responsible for many state-of-the-art sequence modeling results. The main competitor of the LSTM-RNN is the GRU, to be discussed next.
**GRU**
The LSTM architecture is instrumental, but also quite complicated. The complexity of the system makes it hard to analyze and also computationally expensive to work in practice. The gated recurrent unit (GRU) was recently introduced by  (Cho u.a., 2014b)  as an alternative to the LSTM. It was subsequently shown by  (Chung u.a., 2014)to perform comparably to the LSTM on several datasets.

Like the LSTM, the GRU is also based on a gating mechanism, but with substantially fewer gates and without a separate memory component.
$$
\begin{align}
	s_j=R_{GRU}(s_{j-1},x_j)&=(1-z) \odot s_{j-1} +z \odot \widetilde s_j \\&z=\delta (x_j W^{xz}+s_{j-1}W^{sz}) \\ &r=\delta (x_j W^{xr}+s_{j-1}W^{sr}) \\ &\widetilde s_j=tanh(x_j W^{xs}+(r \odot  s_{j-1})W^{sg}) \\ y_j=O_{GRU}&(s_j)=s_j \\s.t.. \ \ s_j, \widetilde s_j \in R^{{d_{s}}},x_i \in R^{d_{x}},z,&r \in R^{d_s},W^{xo} \in R^{d_x \times d_{s}},W^s \in R^{d_s \times d_{s}},W^{so} \in R^{d_s \times d_{s}}
\end{align}
$$
One gate ($r$) is used to control access to t()dhe previous state $s_{j-1}$ and compute a proposed update $\widetilde s_j$. The updated state $s_j$  (which also serves as the output $y_j$ ) is then determined based on an interpolation of the previous state $s_{j-1}$ and the proposal $\widetilde s_j$, where the proportions of the interpolation are controlled using the gate z. 
## Activation Function
There are some major types of neurons that are used in practice that introduce nonlinearities in their computations. As the list below is the most common activation function :
* Sigmoid neurons  (Elfwing u.a., 2017):
$$
	\begin{align}
		\delta(x)= \frac{1}{1+e^{-x}}
	\end{align}
$$
* Tanh neurons  (Nwankpa u.a., 2018):
$$
	\begin{align}
		y=\tag{12}tanh(x)
	\end{align}
$$
* ReLU neurons (Agarap, 2018):
$$
	\begin{align}
		y=max(0,x)
	\end{align}
$$
* Leaky ReLU neurons  (Zhang u.a., 2017):
$$
	\begin{align}
		y=max(0.1x,x)
	\end{align}
$$
* Maxout neurons  (Goodfellow u.a., 2013):
$$
	\begin{align}
		y=max(w^T_1x+b_1,w^T_2x+b_2)
	\end{align}
$$
* ELU neurons s (Clevert u.a., 2015):
$$
	\begin{align}
		y=\begin{cases} x &x\ge 0 \\ \alpha(e^x-1) & x<0 \end{cases}
	\end{align}
$$

## Cost Function and Regularization
An important aspect of the design of a deep neural network is the choice of the cost function. Fortunately, the cost functions for neural networks are more or less the same as those for other parametric models, such as linear models.
In most cases, our parametric model defines a distribution $p(y|;\theta)$, and we use the principle of maximum likelihood. This means we use the cross-entropy between the training data and the model's predictions as the cost function.

Let $y=y_{[1]}\cdot\cdot\cdot\cdot y_{[n]}$ be a vector representing the true multinomial distribution over the labels $1.....n$, and let $\hat y= \hat y_{[1]}\cdot \cdot\cdot\cdot \hat y_{[n]}$ be the linear classifier’s output, which was transformed by the softmax function, and represent the class membership conditional distribution $\hat y_{[i]}=P(y=i|x)$ . The categorical cross entropy loss measures the dissimilarity between the true label distribution $y$ and the predicted label distribution $\hat y$ , and is defined as cross entropy (Amos u. Yarats, 2019):
$$
\begin{align}
	L_{cross-entropy}(\hat y,y)=-\sum_{i} {y_{[i]}log(\hat y_{[i]})}
\end{align}
$$
The total cost function used to train a neural network will often combine one of the primary cost functions described here with a regularization term to avoid overfitting issues.
$$
\begin{align}
	\hat\Theta={\underset {\Theta}{\operatorname {arg\,min} }}\,L(\Theta)+\lambda (R(\Theta))\\ \ \ \ ={\underset {\Theta}{\operatorname {arg\,min} }}\,\frac{1}{n}\sum_{i=1}^{n}L(f(x_i;\Theta))+\lambda R(\Theta)
\end{align}
$$
The weight decay approach  (Nakamura u. Hong, 2019) used for linear models is also directly applicable to deep neural networks and is among the most popular regularization strategies  (Muruganu. Durairaj, 2017).
## Optimization
In order to train the model, we need to solve the optimization problem. A standard solution is to use a gradient-based method. Roughly speaking, gradient-based methods work by repeatedly computing an estimate of the loss $L$ over the training set, computing the gradients of the parameters ‚ with respect to the loss estimate, and moving the parameters in the opposite directions of the gradient. The different optimization methods differ in how the error stimulate is computed, and how “moving in the opposite direction of the gradient” is defined.
While the SGD algorithm can and often does produce good results, more advanced algorithms are also available. The SGD+Momentum  (Duda, 2019) and Nesterov Momentum algorithms (Botev u.a., 2016) are variants of SGD in which previous gradients are accumulated and affect the current update. Adaptive learning rate algorithms, including AdaGrad (Ward u.a., 2018), AdaDelta  (Zeiler, 2012), RMSProp  (Ruder, 2016), and Adam are designed to select the learning rate for each minibatch, sometimes on a per-coordinate basis, potentially alleviating the need of fiddling with learning rate schedules.
## Back-propagation
In the Deep Learning regime, we assumed the function is differentiable, and we can explicitly compute its derivative. In practice, a neural network function consists of many tensor operations chained together, each of which has a simple, known derivative.

Applying the chain rule to the computation of the gradient values of a neural network gives rise to an algorithm called Backpropagation (also sometimes called reverse-mode differentiation). Back-propagation starts with the final loss value and works backward from the top layers to the bottom layers, applying the chain rule to compute the contribution that each parameter had in the loss value.

Nowadays, people will implement networks in modern frameworks that are capable of symbolic differentiation, such as TensorFlow (Abadi u.a., 2016) and PyTorch. This means that, given a chain of operations with a known derivative, they can compute a gradient function for the chain (by applying the chain rule) that maps network parameter values to gradient values. When we have access to such a function, the backward pass is reduced to a call to this gradient function.

## Hyper-parameters
Deep learning (DL) systems expose many tuning parameters (“hyper-parameters”) that affect the performance and accuracy of trained models. Increasingly users struggle to configure hyperparameters, and a substantial portion of time is spent tuning them empirically. Here we review the most critical part of the hyper-parameters.

**Mini-Batch Gradient Descent Hyperparameters**
First, we describe the hyperparameters of mini-batch gradient descent  (Peng u.a., 2019), which updates the network's parameters using gradient descent on a subset of the training data (which is periodically shuffled, or assumed infinite). We'll define the $t_{th} < T$ mini-batch gradient descent update of network parameters ${\theta}$ as :
$$
\begin{align}
	\theta^{(t)} \leftarrow \theta^{(t-1)} -\epsilon_t\frac {1}{B}\sum_{t'=Bt+1}^{B(t+1)}\frac{\partial L(z_{t'},\theta)}{\partial \theta }
\end{align}
$$
where  $z_{t'}$ is example  $t'$  in the training set and the hyperparameters are the loss function $L$, the learning rate at step $t$ $\epsilon _t$, the mini-batch size $B$, and the number of iterations $$T$$. Note that we must define $\theta ^0$, which is also a hyperparameter. For a specific optimization method, as we describe the momentum, it helps to “smooth” the gradient updates using a leaky integrator filter with parameter $\beta$ by
$$
\begin{align}
	\bar g \leftarrow (1-\beta)\bar g+ \beta \frac{\partial L(z_{t'},\theta)}{\partial \theta } 
\end{align}
$$
$\bar g$ can then be used in place of the “true” gradient update in gradient descent. Some mathematically motivated approaches can ensure much faster convergence when using appropriate momentum; however, for pure stochastic gradient descent, standard gradient updates $\beta =1$  with a harmonically decreasing learning rate is optimal.
**Model Hyperparameters**
The structure of the neural network itself involves numerous hyperparameters in its design, including the size and nonlinearity of each layer. The numeric properties of the weights are often also constrained in some way, and their initialization can have a substantial effect on model performance. Finally, the preprocessing of the input data can also be essential for ensuring convergence. As a practical note, many hyperparameters can vary across layers.


* Number of hidden units
* Weight decay, the purpose is to reduce overfitting with the regularization term.
* Activation sparsity, it may be advantageous for the hidden unit activations to be sparse. We should consider the type of penalty, L1, L2, or combined L1 and L2.
* Nonlinearity, we also should select the type of activation function, as we have described in the previous chapter.
* Weight initialization, biases are typically initialized to 00, but weights must be initialized carefully. Their initialization can have a significant impact on the local minimum found by the training algorithm.
* Random seeds  (Colas u.a., 2018) and model averaging. Many of the processes involved in training a neural network involve using a random number generator (e.g., random sampling of training data, weight initialization.). As a result, the seed passed to the random number generator can have a slight effect on the results. However, a different random seed can produce a non-trivially different model (even if it performs about as well). As a result, it is common to train multiple models with multiple random seeds and use model averaging (bagging, Bayesian methods) to improve performance.
* Preprocess input data, the statistics of the input data can have a substantial effect on network performance. Element-wise standardization (subtract the mean and divide by the standard deviation), Principal Component Analysis, uniformization (transform each feature value to its approximate normalized rank or quantile), and nonlinearities such as the logarithm or square root are common.

**Hyperparameters Space Exploration**
The number of hyperparameters delineated above indicates that there are a substantial number of choices to be made when creating a neural network learner and that these choices will affect the success and failure of the model. In order to ensure reproducibility of results, a principled approach should be used for setting hyperparameters, or, at the very least, they should be explicitly stated in the model description.

There is some suggestion we can take when we explore the hyperparameters, including coordinate descent, grid search, and random search.

# Experiments Setup
## Tensorflow and Keras
In our experiment, we use Tensorflow and Keras to build and train the model. Tensorflow is a numeric computing framework and machine learning library. It is the most famous framework for the implementation of Deep Learning. Keras is a high-level neural networks API, written in Python and capable of running on top of TensorFlow, CNTK, or Theano. It was developed with a focus on enabling fast experimentation.  Being able to go from idea to result with the least possible delay is key to doing proper research.

## feature select
In the initial version of the data, we have collected 48 features for each stock. In the experiment stage, we find that the number of features does not affect the accuracy too much and reduce to 10 features as we described the dataset above.

To have the intuitive sense of the dataset, here we post a data fragment for the stock code sh600033 in Table1.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-3560d280d679c2e3.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

Each row represents one day in one market day, columns from left to right are the date, stock code name, opening price, day high price, day low price, close price, changes, volumes, five days moving average, ten days moving average and twenty days moving average.

## Label Design
We divide the experiment into two different problems, binary classification and multiclass classification problem.

In our binary classification problem, we design the output as two output labels. Label $0$ represent the stock in which the price rose, and label $1$ represent the stock price fell. Out network always be ended with a Dense layer with one unit and a sigmoid activation. The output of our network is a scalar between 0 and 1, encoding a probability.

In our multiclass classification problem, we design the output as 8 classes and a single label. Our network ended with a softmax activation so that it will output a probability distribution over the 8 output classes. Categorical crossentropy is the default loss function for our multiclass classification problem. It minimizes the distance between the probability distributions output by the network and the true distribution of the targets. In the Chinese stock market, the maximum daily increase of each stock is 10% and the maximum decline is also 10%. We divide the change from $-0.1$ to $0.1$ into 8 classes.
$$
\begin{align}
	&Class1 \in [-0.1,-0.075) \\ &Class2 \in [-0.075,-0.05) \\&Class3 \in [-0.05,-0.025)\\&Class4 \in [-0.025,0)\\&Class5 \in [0,0.025)\\&Class6 \in [0.025,0.05)\\&Class7 \in [0.05,0.075)\\&Class8 \in [0.075,0.1] 
\end{align}
$$
## Model design
we implement the experiment by considering various networ architectures, including MLP, CNN, SimpleRNN, GRU, and LSTM.
* In MLP, we mainly use 512 neurons units and following the dropout layer with 0.5 to drop. We also considered different numbers of dense layers in our design.
* For CNN, we use the 1D convolution neural layer, with the different filters, kernel size, strides.
* For SimpleRNN, LSTM, and GRU: we try different units and activation functions.
* For BIRNN, we use the bi-direction trick for all the RNN models.
## Main hyperparameter design
**Learning rate:** In our experiment, the RMSprop optimizer is the default choice. In some cases, we select some specific learning rate for the RMSprop optimizer, but we let it automated running for most of the cases. We also used the callback function to reduce the learning rate when the validation loss has stopped improving.

**Activation function:** we only consider ReLU and tanh as our activation function pool in the nonoutput layers. In the output layer, we use sigmoid for the binary classification problem and  for the multiclass classification problem.

**Epochs:** we set the maximum epochs as 200 and use the early stop trick to interrupt training once the target metric being monitored has stopped improving for a various number of epochs.

**Lookback and delay steps:** The exact formulation of the stock prediction problem will be given data going as far back as lookback timesteps ( a timestep is one trading day ), we set the lookback trading days as 20 and consider the output in one delayed trading day.

**Batch size:** we set the batch size as 1024 in our whole experiment.

**Missing data process:**  Normally, there are many tricks for missing data process method. For example, fill with average value, last value, min value, max value, or zero. In our experiment, we just fill the missing data as zero.

We list all the architecture in our experiment in Table2.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-4887fa84a558395c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

In order to have an intuitive understanding of the network architecture, We report the architecture of network number 21 from Table2  and show the detail in Table 3.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-176db5372444fcb9.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

## Environment
Most modern neural network implementations are based on graphics processing units(GPU). GPU is specialized hardware components that were originally developed for graphics applications, but the rise of GPU also dramatically improve the efficiency of neural network training  (Wang u.a., 2019).Nevertheless, in the daily applications, using GPU is a very high-cost solution, so we also investigated the case of training small neural networks on the CPU. We create some VMs for theses case with the specifications as 6 vCPU and 24G memory.

# Result and Analysis
Our experiment result performance is compared with the baseline models. We achieve 0.319 accuracies in the eight classes classification task and achieve 0.686 accuracies in the binary classification task.

In our research work, we have tried more than 200 test cases with different architecture, hyperparameters, and output design. We also give a brief overview of the CPU running time for each model.

## Architecture Compare}
From the result of the experiment, we find in our designed architecture, and there is a tiny difference in training accuracy, validation accuracy, and test accuracy in the different architecture. We showed the result of the different architecture in Table4, and Table5.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-ce4f370d41e9039b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/15463866-532be8a1b50015aa.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
From the result of the above tables, we observed that:
* The accuracy in the various model only has a tiny difference.
* RNN series model, include LSTM, GRU has regularly outperformed the other architecture.
* The type of activation does not affect the result, but it impacts the learning progress in the experiment. We find that the ReLU activation function will lead to a smooth learning curve, and tanh activation will lead to a jittery learning curve. Fig 6 shows a comparison between the two types of activation function.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-b5e33ef8a313b285.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

* the learning rate does not impact the result as we use the RMSprop as the default optimizer.

## Computation Time in CPU
We have created more than 30 VMs with the specifications as 6 vCPU and 24G memory to test the running time per epoch in every architecture. The result showed that different network architectures have considerable differences in CPU running time. In practice, we suggest selecting the simpler architecture and only loss a little accuracy of the result.
We report the best top three inTable 6 and the worst top three in Table 7.
![image.png](https://upload-images.jianshu.io/upload_images/15463866-d08ce7a3982b6e9e.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![image.png](https://upload-images.jianshu.io/upload_images/15463866-865c8a36e338f148.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
# Discussion and Future Work
Focusing on the stock market in China, we propose the basic Deep Neural Network to get a comparable result for the stock prediction problem. According to the competent performance of models in the practical application, deep neural network methods have strong implications for investors as well as the entire stock market in China. The financial industry might integrate deep neural network prediction result in traditional prediction models to make better decisions. This is an interesting filed and promising direction in behavioral finance.

This study has inevitable limitations, which might be exciting directions in future work. We prepare to explore some potential directions to improve accuracy.

First, we could implement different structure of feature extractor, such as Transformer  (Jaderbergu.a., 2015), BERT(Devlin u.a., 2018), GAN(Krizhevsky u.a., 2012).

Second, other modern neural network architecture may be worth to try like AlexNet(Creswell u.a.,2017), GoogLeNet(Szegedy u.a., 2014), VGG19(Simonyan u. Zisserman, 2017), ResNet(He u.a.,2015).

Third, the attention mechanism  (Liu u. Wang, 2019) can be introduced to handle long-term dependency, which cannot be handled by RNN.

Forth, our hyper-parameter is only designed by heuristic, and if we have the formal policy for hyperparameter search, the result should be better than the current result. The coordinate descent, grid search, and random search policy are the suggested method for the future hyperparameters fine-tune.

Fifth, the deep reinforcement learning method is also well suited for the stock market prediction problem.

Sixth, the feature of the current dataset should not be enough if we use a robust network architecture; we should consider more features for each stock. Furthermore, some other conditions should also be beneficial, such as the international market trends, the future market price, the exchange market price, and the emotion status in the social network.

#REFERENCES
TensorFlow: Large-Scale Machine Learning on Heterogeneous Distributed Systems. In: CoRR abs/1603.04467 (2016). http://arxiv.org/abs/1603.04467

Deep Learning using Rectified Linear Units (ReLU). In: CoRR abs/1803.08375 (2018). http://arxiv.org/abs/1803.08375

The Differentiable Cross-Entropy Method. In: CoRR abs/1909.12830 (2019). http://arxiv.org/abs/1909.12830

A hybrid genetic-neural architecture for stock indexes forecasting. In: Inf. Sci. 170 (2005), Nr. 1, 3–33. http://dx.doi.org/10.1016/j.ins.2003.03.023 . – DOI 10.1016/j.ins.2003.03.023

Forecasting stock market short-term trends using a neuro-fuzzy based methodology. In: Expert Syst. Appl. 36 (2009), Nr. 7, 10696–10707. http://dx.doi.org/10.1016/j.eswa.2009. 02.043. – DOI 10.1016/j.eswa.2009.02.043

Twitter mood predicts the stock market. In: J. Comput. Science 2 (2011), Nr. 1, 1–8. http://dx.doi.org/10.1016/j.jocs.2010.12.007. – DOI 10.1016/j.jocs.2010.12.007 Nesterov’s Accelerated Gradient and Momentum as approximations to Regularised Update Descent. In: CoRR abs/1607.01981 (2016). http://arxiv.org/abs/1607.01981

Support vector machine with adaptive parameters in financial time series forecasting. In: IEEE Trans. Neural Networks 14 (2003), Nr. 6, 1506–1518. http://dx.doi.org/10.1109/TNN. 2003.820556. – DOI 10.1109/TNN.2003.820556

Exchange rate prediction redux: new models, new data, new currencies. In: Journal of International Money and Finance 95 (2019), S. 332–362

Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. In: Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing, EMNLP 2014, October 25-29, 2014, Doha, Qatar, A meeting of SIGDAT, a Special Interest Group of the ACL, 2014, S. 1724–1734

Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation. In: CoRR abs/1406.1078 (2014). http://arxiv.org/abs/1406.1078

Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling. In: CoRR abs/1412.3555 (2014). http://arxiv.org/abs/1412.3555

Fast and Accurate Deep Network Learning by Exponential Linear Units (ELUs). In: Computer Science (2015)

How Many Random Seeds? Statistical Power Analysis in Deep Reinforcement Learning Experiments.In: CoRR abs/1806.08295 (2018). http://arxiv.org/abs/1806.08295

Generative Adversarial Networks: An Overview. In: CoRR abs/1710.07035 (2017). http://arxiv.org/abs/1710.07035

BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. In: CoRR abs/1810.04805 (2018). http://arxiv.org/abs/1810.04805

SGD momentum optimizer with step estimation by online parabola model. In: CoRR abs/1907.07063(2019). http://arxiv.org/abs/1907.07063

Sigmoid-Weighted Linear Units for Neural Network Function Approximation in Reinforcement Learning. In: CoRR abs/1702.03118 (2017). http://arxiv.org/abs/1702.03118

Finding Structure in Time. In: Cognitive Science 14 (1990), Nr. 2, 179–211.  http://dx.doi.org/10.1207/s15516709cog1402_1. – DOI 10.1207/s15516709cog1402_1

Deep Learning. MIT Press, 2016 (Adaptive computation and machine learning). http://www.deeplearningbook.org/. – ISBN 978–0–262–03561–3

Maxout Networks. In: CoRR abs/1302.4389 (2013). http://arxiv.org/abs/1302.4389 Deep Residual Learning for Image Recognition. In: CoRR abs/1512.03385 (2015). http://arxiv.org/abs/1512.03385

Hierarchical Recurrent Neural Networks for Long-Term Dependencies. In: Advances in Neural Information Processing Systems 8, NIPS, Denver, CO, USA, November 27-30, 1995, 1995, S.493–499

Long short-term memory. In: Neural computation 9 (1997), Nr. 8, S. 1735–1780 Forecasting stock markets using wavelet transforms and recurrent neural networks: An integrated system based on artificial bee colony algorithm. In: Appl. Soft Comput. 11 (2011),Nr. 2, 2510–2525. http://dx.doi.org/10.1016/j.asoc.2010.09.007 . – DOI 10.1016/j.asoc.2010.09.007

Application of wrapper approach and composite classifier to the stock trend prediction. In: Expert Syst. Appl. 34 (2008), Nr. 4, 2870–2878. http://dx.doi.org/10.1016/j.eswa.2007.05.035. – DOI 10.1016/j.eswa.2007.05.035

Spatial Transformer Networks. In: CoRR abs/1506.02025 (2015). http://arxiv.org/abs/1506.02025

Modelling high-frequency limit order book dynamics with support vector machines. In: Quantitative Finance 15 (2015), Nr. 8, S. 1–15

Financial time series forecasting using support vector machines. In: Neurocomputing 55 (2003),Nr. 1-2, 307–319. http://dx.doi.org/10.1016/S0925-2312(03)00372-2 . – DOI10.1016/S0925–2312(03)00372–2

An intelligent hybrid trading system for discovering trading rules for the futures market using rough sets and genetic algorithms. In: Appl. Soft Comput. 55 (2017), 127–140. http://dx.doi.org/10.1016/j.asoc.2017.02.006. – DOI 10.1016/j.asoc.2017.02.006

ImageNet Classification with Deep Convolutional Neural Networks. In: Advances in Neural Information Processing Systems 25: 26th Annual Conference on Neural Information Processing Systems 2012. Proceedings of a meeting held December 3-6, 2012, Lake Tahoe, Nevada, United States, 2012, S. 1106–1114

Inferring trade direction from intraday data. In: The Journal of Finance 46 (1991), Nr. 2, S. 733–746 Forecasting stock indices: a comparison of classification and level estimation models. In: International Journal of Forecasting 16 (2000), Nr. 2, S. 173–190

Empirical analysis: stock market prediction via extreme learning machine. In: Neural Computing and Applications 27 (2016), Nr. 1, 67–78. http://dx.doi.org/10.1007/s00521-014-1550-z. – DOI 10.1007/s00521–014–1550–z

Hybrid Neural Networks for Learning the Trend in Time Series. In: Proceedings of the Twenty-Sixth International Joint Conference on Artificial Intelligence, IJCAI 2017, Melbourne, Australia, August
19-25, 2017, 2017, S. 2273–2279

A Numerical-Based Attention Method for Stock Market Prediction With Dual Information. In: IEEE Access 7 (2019), 7357–7367. http://dx.doi.org/10.1109/ACCESS.2018.2886367 .
– DOI 10.1109/ACCESS.2018.2886367

Regularization and Optimization strategies in Deep Convolutional Neural Network. In: CoRRabs/1712.04711 (2017). http://arxiv.org/abs/1712.04711

Adaptive Weight Decay for Deep Neural Networks. In: CoRR abs/1907.08931 (2019). http://arxiv.org/abs/1907.08931

Activation Functions: Comparison of trends in Practice and Research for Deep Learning. In: CoRRabs/1811.03378 (2018). http://arxiv.org/abs/1811.03378

Understanding the exploding gradient problem. In: CoRR abs/1211.5063 (2012). http://arxiv.org/abs/1211.5063

Accelerating Minibatch Stochastic Gradient Descent using Typicality Sampling. In: CoRR abs/1903.04192 (2019). http://arxiv.org/abs/1903.04192

An overview of gradient descent optimization algorithms. In: CoRR abs/1609.04747 (2016).http://arxiv.org/abs/1609.04747

Textual analysis of stock market prediction using breaking financial news: The AZFin text system.In: ACM Trans. Inf. Syst. 27 (2009), Nr. 2, 12:1–12:19. http://dx.doi.org/10.1145/1462198.1462204. – DOI 10.1145/1462198.1462204

Bidirectional recurrent neural networks. In: IEEE Transactions on Signal Processing 45 (1997), Nr.11, S. 2673–2681

Very Deep Convolutional Networks for Large-Scale Image Recognition. CoRR. 2014;: abs/1409.1556.In: arXiv preprint arXiv:1409.1556 (2017)

Going Deeper with Convolutions. In: CoRR abs/1409.4842 (2014). http://arxiv.org/abs/1409.4842

Stock price direction prediction by directly using prices data: an empirical study on the KOSPIand HSI. In: IJBIDM 9 (2014), Nr. 2, 145–160. http://dx.doi.org/10.1504/IJBIDM.2014.065091. – DOI 10.1504/IJBIDM.2014.065091

Market Index and Stock Price Direction Prediction using Machine Learning Techniques: An empiricalstudy on the KOSPI and HSI. In: CoRR abs/1309.7119 (2013). http://arxiv.org/abs/1309.7119

Benchmarking TPU, GPU, and CPU Platforms for Deep Learning. In: CoRR abs/1907.10701 (2019).http://arxiv.org/abs/1907.10701

AdaGrad stepsizes: Sharp convergence over nonconvex landscapes, from any initialization. In: CoRRabs/1806.01811 (2018). http://arxiv.org/abs/1806.01811

Recent Trends in Deep Learning Based Natural Language Processing. In: CoRR abs/1708.02709(2017). http://arxiv.org/abs/1708.02709

ADADELTA: An Adaptive Learning Rate Method. In: CoRR abs/1212.5701 (2012). http://arxiv.org/abs/1212.5701

Combining News and Technical Indicators in Daily Stock Price Trends Prediction. In: Advances in Neural Networks - ISNN 2007, 4th International Symposium on Neural Networks, ISNN 2007,Nanjing, China, June 3-7, 2007, Proceedings, Part III, 2007, S. 1087–1096

Dilated convolution neural network with LeakyReLU for environmental sound classification. In:2017 22nd International Conference on Digital Signal Processing (DSP), 2017

Financial prediction using neural networks. International Thomson Computer Press, 1996
