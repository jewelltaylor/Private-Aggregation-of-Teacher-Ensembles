
# Introduction 

The following notebook provides a comprehensive overview and implementation of the [Private Aggregation of Teacher Ensembles (PATE)](https://arxiv.org/pdf/1610.05755.pdf) using [PyTorch](https://pytorch.org/) and [PySyft](https://github.com/OpenMined/PySyft). The dataset that is used is [Home Credit Default Dataset](https://www.kaggle.com/c/home-credit-default-risk/overview) which contains features about credit applicants drawn from the internal operations of a financial insititution as well as corresponding Credit Bureau data. The target is whether or not the appliant will default on their debt. The following sections contain a detailed a theoretical overview and subsequent implementation.

# Background

## Private AI

### Introduction to Private AI
A major concern of production machine learning systems is maintaining the privacy of the data that was used to train it. This is especially relevant in use cases with private and sensitive information. In the Private AI literature, a party looking to reveal information about the data that a model is trained on is called an adversary. Even if the adversary does not have access to the data explicitly, access to the model and/or its outputs can reveal information about the data that the model was trained on.

### Privacy Attacks
An adversary employs a privacy attack in order to reveal information about the data that the model was trained on. Two attacks that are well researched in the Private AI Literature: 
- **Model Inversion Attack:** Try to obtain information about typical samples in the training dataset. This involves attacks that euther reconst a specific sample or representative samples in the training dataset. 

- **Membership Inference Attack:** Try to ascertain whether or not a sample was used to train a model. 

### Threat Models
Each adversary has a threat model that informs the manner in which they go about a privacy attack. The threat model describes the level of access that an adverary has to a model. At a high level, there are two threat models: 
- **Black Box Adversary:** Solely able to query the model. Thus, the only information made available to the adversary is the inputs and outputs of the model. 
- **White Box Adversary:** The adversary is able to query the model as well as have access to its internal parameters. This implies that the adversary has access to the input, output and the intermediate computations made by the model. 

## Differential Privacy 
### Introduction to Differential Privacy
In order to evaluate the robustness of a model to privacy attacks, we have to define a framework through which we can obtain a quantitative measure that describes its performance in terms of privacy. To this end, Differential Privacy has been proposed and offers a powerful mechanism to assess and rank the privacy of models. It does so based on the sensitivity of a model to the inclusion of a specific sample.

<p align="center">
<img width="433" alt="Screen Shot 2021-09-28 at 5 41 20 PM" src="https://user-images.githubusercontent.com/34798787/137939981-a9968386-4a28-4447-ad9e-5babab497eeb.png">  
</p>
<center>
<a href=https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-privacy-preserving-data-analysis-introduction-our>Source</a>  
</center>


Specified formally, the definition of differential privacy is given by the below inequality: 
<p align="center">
<img width="185" alt="Screen Shot 2021-10-19 at 11 40 16 AM" src="https://user-images.githubusercontent.com/34798787/137944948-cffd9db8-5a29-4f98-bf5a-b5b7d4bbe25b.png">
</p>

where M is the model, x is orginal the dataset, x' is the original dataset augmented to include or exclude a single sample. $\epsilon$ acts as a metric of privacy loss based on a differential change in data. The smaller the value is, the better privacy protection. Rearranging the inequality yields:

<p align="center">
<img width="345" alt="Screen Shot 2021-10-19 at 3 37 50 PM" src="https://user-images.githubusercontent.com/34798787/137978800-d3c56d46-5b7c-4b37-a61e-6f7b8daa61d2.png">
</p>
This is a strong to guarentee to achieve in practice so a failure probability in order to relax this constraint we add a failure probability, $\delta$, to the RHS of the inequality: 

<p align="center">
<img width="400" alt="Screen Shot 2021-10-19 at 3 35 50 PM" src="https://user-images.githubusercontent.com/34798787/137978567-7a0313bc-d9cd-4a0d-9f9b-7df3711f2c3d.png">
</p>

As long as delta is smaller than the probability that a sample occurs in the dataset, we will still obtain a high degree of privacy. This allows us to relax the guarentee we need to provide while maintaining an acceptable level of privacy. As parameters that define the interval of differential privacy, a model with $\epsilon$ and $\delta$ is ($\epsilon$, $\delta$)-differentially private. 

### Composition in Differential Privacy
An important consideration in practice is determining the differential privacy of a composition of a number of models. Drawing from [The Algorithmic Foundations
of Differential Privacy](https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf), the composition of k differentially private mechanisms, where the $i$th mechanism is ($\epsilon_{i}$, $\delta_{i}$)-differentially private, for 1 $\leq$ i $\leq$ k, is ($\sum \epsilon_{i}$, $\sum \delta_{i}$)-differentially private.

### Privacy Amplication Theorem
If we select a subset of samples from the dataset, it is intuitive that we would incur a privacy loss that is lower than that of using the entire dataset. This is the essence of the Privacy Amplication Theorem which states that if we randomly sample a q of the data, rather than the entrie dataset, then an ($\epsilon$, $\delta$) private mechansm becomes ($q\epsilon$, $q\delta$)-differentially private.

### Fundamental Law of Information Recovery
The Fundamental Law of Information Recovery states that overly accurate estimates of too many statistics erodes the privacy of data. This implies that continuously querying a private mechanism will increase the privacy loss incurred. 

### Implications for Training and Testing Machine Learning Models
Within the context of training and testing a machine learning model, we can use Composition in Differentially Privacy and Privacy Amplication Theorem to derive the privacy loss in situations where we are iteritvely querying a model with a random subset of the data, such as in the case of training and testing Neural Networks. The summation of privacy losses over queries follows the intuition from the Fundamental Law of Information Recovery that continuously querying a private mechanism will increase the privacy loss incurred. 

## Differentially Private Stochastic Gradient Descent (DP-SGD)

### Introduction to DP-SGD
The seminal paper that applies Differential Privacy to Deep Learning is [Deep Learing with Differential Privacy](https://arxiv.org/abs/1607.00133). In this paper, a differentially-private variant of stochastic gradient descent is proposed (DP-SGD). 

<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/138136971-c51261d3-73de-4afb-97b9-71e149ddfacd.png">
</p>

<center>
<a href=https://secml.github.io/class4>Source</a>  
</center>

### Estimating Differential Privacy for a Single Gradient Update
As in vanilla SGD, we start with calculating the gradient of a batch of data. In order to limit the amount information we learn from the batch, we clip the gradient at C which is a hyperpareter for the algorithim. We than add noise $\sigma^2$ proportional C and than update the parameters of the model. 

### Estimating Differential Privacy for accross Gradient Updates
In order to get the total privacy loss, we must aggregate the privacy loss for each gradient update. Based on Composition in Differential Privacy and The Privacy Amplicaiton Theorem, we can define a Naive estimate of the upper bound of the privacy loss. Given an ($\epsilon$, $\delta$)-differentially private algorithim, with a batch size proportional to $q$ run for T iterations has a differential privacy of ($Tq\epsilon$, $Tq\delta$). 

It turns out that a more complex analysis can yield lower bounds for the privacy loss. Thus, finding the lowest bound possible allows us to more accurately determine the privacy loss and to avoid overstimating it. For example, the Strong Compositon Theorem can be used to prove an even lower bound on the privacy loss. In the DP-SGD paper, the authors propose the Moments Accountant which provided the lowest bound 
- **Naive Analysis:** ($Tq\epsilon$, $Tq\delta$)-differentially private
- **Strong Composition Theorem:** ($O(q\epsilon\sqrt{Tlog(1/\delta)})$, $Tq\delta$)-differentially private
- **Moments Accountant:** ($O(q\epsilon\sqrt{T})$, $\delta$)-differentially private.

### Moments Accountant
The fundamental insight of the Moments Accountant technique is that the privacy loss is a random variable. Thus, if we look at the distribution of the privacy loss, we can see that $\epsilon$ defines the privacy budget for the loss and $\delta$ provides an upper bound on the tail of the distribution. 

<p align="center">
<img width="500" alt="Screen Shot 2021-10-20 at 1 50 02 PM" src="https://user-images.githubusercontent.com/34798787/138145128-0d67885f-a5c4-448f-9fcd-78fa0452a3ce.png">
</p>
<center>
<a href=https://www.youtube.com/watch?v=jm1Sfdno_5A>Source</a>  
</center>

By treating the privacy loss as a random variable, we can leverage probability theory to derive a lower bound on the privacy loss. Specifically, the Moments Accountant Technique uses the moments of the distribution (ie mean, variance, skewness and kurtosis) in order to do so. Incorporating the higher order information of the distibution made available through the moments  allows us to derive a lower bound on the privacy loss: $O(q\epsilon\sqrt{T})$, $\delta$). For a deeper look into the Moments Accountant technique, refer to [Deep Learing with Differential Privacy](https://arxiv.org/abs/1607.00133).

## Private Aggregation of Teacher Ensembles (PATE)

### Introduction to PATE
In [Scalable Private Learning with PATE](https://arxiv.org/abs/1802.08908), authors set out solve the problem of preserving the privacy of training data when training classfiers. They began with defining certain criteria for the solution:
- Differential privacy protection guarentees
- Intuitive privacy protection guarentees 
- **Independent of learning algorithim**

<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/139095511-ff27899c-80f0-45bf-a4d0-d18820c20e34.png">
</p>

<center>
<a href=https://www.youtube.com/watch?v=cjo_u_yT2wQ&t=1s>Source</a>  
</center>

### Ensembling

In differential privacy, we seek to learn general trends from the data. That is, the outcome of the the prediction of a sample, should be the same whether or not we choose to include an sample in the dataset for each sample in the dataset. One natural way to achieve this to use an ensemble of models trained on random subsets of the dataset. In this way, the prediction for a specific sample is less likely to depend on a single example. Furthermore, if there is a strong consensus among the ensemble, it is likely the prediction stems from a general trend rather than a specific sample. In practice, this allows us to define a lower, data dependent bound on the privacy loss. 

Although ensembling enhances differentially privacy, it is still possible that the inclusion/exclusion of samples in the various teachers will change the outcome of the predictions. This is especially the case when there is not a strong consensus among the ensemble; the vote of single teacher can sway the prediction of the output. 

### Noisy Aggregation

In order to address the aforementioned shortcoming, we can aggregate the votes of the teachers in a noisy way. This is realized by adding noise to the final prediction.

<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/139089966-fa783578-cc52-4a58-bcb4-9a06432868ac.png">
</p>
<center>
<a href=https://www.youtube.com/watch?v=cjo_u_yT2wQ&t=1s>Source</a>  
</center>

### Student Training
Although the aggregated teacher is a good step towards differential privacy, it does have some shortcomings: 
- Each prediction increases total privacy loss. Thus, the privacy budget creates a tradeoff between the accuracy and number of predictions.
- Inspection of internal may reveal private data. However, we want privacy guarentees that hold for white box adversaries. 

As a result, the PATE framework introduces a student network that is trained on publicly available data using labels from Teacher Model.

<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/138165916-a61f044d-4d45-4f5f-8e62-3fca94b65b84.png">
</p>
<center>
<a href=https://www.youtube.com/watch?v=cjo_u_yT2wQ&t=1s>Source</a>  
</center>






