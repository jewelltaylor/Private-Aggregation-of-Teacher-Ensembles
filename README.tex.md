
<!DOCTYPE html>
<html>

<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Welcome file</title>
  <link rel="stylesheet" href="https://stackedit.io/style.css" />
</head>

<body class="stackedit">
  <div class="stackedit__html"><h1 id="introduction">Introduction</h1>
<p>The following notebook provides a comprehensive overview and implementation of the <a href="https://arxiv.org/pdf/1610.05755.pdf">Private Aggregation of Teacher Ensembles (PATE)</a> using <a href="https://pytorch.org/">PyTorch</a> and <a href="https://github.com/OpenMined/PySyft">PySyft</a>. The dataset that is used is <a href="https://www.kaggle.com/c/home-credit-default-risk/overview">Home Credit Default Dataset</a> which contains features about credit applicants drawn from the internal operations of a financial insititution as well as corresponding Credit Bureau data. The target is whether or not the appliant will default on their debt. The following sections contain a detailed a theoretical overview and subsequent implementation.</p>
<h1 id="background">Background</h1>
<h2 id="private-ai">Private AI</h2>
<h3 id="introduction-to-private-ai">Introduction to Private AI</h3>
<p>A major concern of production machine learning systems is maintaining the privacy of the data that was used to train it. This is especially relevant in use cases with private and sensitive information. In the Private AI literature, a party looking to reveal information about the data that a model is trained on is called an adversary. Even if the adversary does not have access to the data explicitly, access to the model and/or its outputs can reveal information about the data that the model was trained on.</p>
<h3 id="privacy-attacks">Privacy Attacks</h3>
<p>An adversary employs a privacy attack in order to reveal information about the data that the model was trained on. Two attacks that are well researched in the Private AI Literature:</p>
<ul>
<li>
<p><strong>Model Inversion Attack:</strong> Try to obtain information about typical samples in the training dataset. This involves attacks that euther reconst a specific sample or representative samples in the training dataset.</p>
</li>
<li>
<p><strong>Membership Inference Attack:</strong> Try to ascertain whether or not a sample was used to train a model.</p>
</li>
</ul>
<h3 id="threat-models">Threat Models</h3>
<p>Each adversary has a threat model that informs the manner in which they go about a privacy attack. The threat model describes the level of access that an adverary has to a model. At a high level, there are two threat models:</p>
<ul>
<li><strong>Black Box Adversary:</strong> Solely able to query the model. Thus, the only information made available to the adversary is the inputs and outputs of the model.</li>
<li><strong>White Box Adversary:</strong> The adversary is able to query the model as well as have access to its internal parameters. This implies that the adversary has access to the input, output and the intermediate computations made by the model.</li>
</ul>
<h2 id="differential-privacy">Differential Privacy</h2>
<h3 id="introduction-to-differential-privacy">Introduction to Differential Privacy</h3>
<p>In order to evaluate the robustness of a model to privacy attacks, we have to define a framework through which we can obtain a quantitative measure that describes its performance in terms of privacy. To this end, Differential Privacy has been proposed and offers a powerful mechanism to assess and rank the privacy of models. It does so based on the sensitivity of a model to the inclusion of a specific sample.</p>
<p align="center">
<img width="433" alt="Screen Shot 2021-09-28 at 5 41 20 PM" src="https://user-images.githubusercontent.com/34798787/137939981-a9968386-4a28-4447-ad9e-5babab497eeb.png">  
</p>
<center>
<a href="https://www.nist.gov/blogs/cybersecurity-insights/differential-privacy-privacy-preserving-data-analysis-introduction-our">Source</a>  
</center>
<p>Specified formally, the definition of differential privacy is given by the below inequality:</p>
<p align="center">
<img width="185" alt="Screen Shot 2021-10-19 at 11 40 16 AM" src="https://user-images.githubusercontent.com/34798787/137944948-cffd9db8-5a29-4f98-bf5a-b5b7d4bbe25b.png">
</p>
<p>where M is the model, x is orginal the dataset, x’ is the original dataset augmented to include or exclude a single sample. <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span> acts as a metric of privacy loss based on a differential change in data. The smaller the value is, the better privacy protection. Rearranging the inequality yields:</p>
<p align="center">
<img width="345" alt="Screen Shot 2021-10-19 at 3 37 50 PM" src="https://user-images.githubusercontent.com/34798787/137978800-d3c56d46-5b7c-4b37-a61e-6f7b8daa61d2.png">
</p>
This is a strong to guarentee to achieve in practice so a failure probability in order to relax this constraint we add a failure probability, $\delta$, to the RHS of the inequality: 
<p align="center">
<img width="400" alt="Screen Shot 2021-10-19 at 3 35 50 PM" src="https://user-images.githubusercontent.com/34798787/137978567-7a0313bc-d9cd-4a0d-9f9b-7df3711f2c3d.png">
</p>
<p>As long as delta is smaller than the probability that a sample occurs in the dataset, we will still obtain a high degree of privacy. This allows us to relax the guarentee we need to provide while maintaining an acceptable level of privacy. As parameters that define the interval of differential privacy, a model with <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span> and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span> is (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private.</p>
<h3 id="composition-in-differential-privacy">Composition in Differential Privacy</h3>
<p>An important consideration in practice is determining the differential privacy of a composition of a number of models. Drawing from <a href="https://www.cis.upenn.edu/~aaroth/Papers/privacybook.pdf">The Algorithmic Foundations<br>
of Differential Privacy</a>, the composition of k differentially private mechanisms, where the <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>i</mi></mrow><annotation encoding="application/x-tex">i</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.65952em; vertical-align: 0em;"></span><span class="mord mathnormal">i</span></span></span></span></span>th mechanism is (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>ϵ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\epsilon_{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.58056em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal">ϵ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msub><mi>δ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\delta_{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.84444em; vertical-align: -0.15em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03785em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>)-differentially private, for 1 <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>≤</mo></mrow><annotation encoding="application/x-tex">\leq</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.77194em; vertical-align: -0.13597em;"></span><span class="mrel">≤</span></span></span></span></span> i <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>≤</mo></mrow><annotation encoding="application/x-tex">\leq</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.77194em; vertical-align: -0.13597em;"></span><span class="mrel">≤</span></span></span></span></span> k, is (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>∑</mo><msub><mi>ϵ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\sum \epsilon_{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.00001em; vertical-align: -0.25001em;"></span><span class="mop op-symbol small-op" style="position: relative; top: -5e-06em;">∑</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal">ϵ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: 0em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mo>∑</mo><msub><mi>δ</mi><mi>i</mi></msub></mrow><annotation encoding="application/x-tex">\sum \delta_{i}</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.00001em; vertical-align: -0.25001em;"></span><span class="mop op-symbol small-op" style="position: relative; top: -5e-06em;">∑</span><span class="mspace" style="margin-right: 0.166667em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span><span class="msupsub"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.311664em;"><span class="" style="top: -2.55em; margin-left: -0.03785em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight"><span class="mord mathnormal mtight">i</span></span></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.15em;"><span class=""></span></span></span></span></span></span></span></span></span></span>)-differentially private.</p>
<h3 id="privacy-amplication-theorem">Privacy Amplication Theorem</h3>
<p>If we select a subset of samples from the dataset, it is intuitive that we would incur a privacy loss that is lower than that of using the entire dataset. This is the essence of the Privacy Amplication Theorem which states that if we randomly sample a q of the data, rather than the entrie dataset, then an (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>) private mechansm becomes (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>q</mi><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">q\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>q</mi><mi>δ</mi></mrow><annotation encoding="application/x-tex">q\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private.</p>
<h3 id="fundamental-law-of-information-recovery">Fundamental Law of Information Recovery</h3>
<p>The Fundamental Law of Information Recovery states that overly accurate estimates of too many statistics erodes the privacy of data. This implies that continuously querying a private mechanism will increase the privacy loss incurred.</p>
<h3 id="implications-for-training-and-testing-machine-learning-models">Implications for Training and Testing Machine Learning Models</h3>
<p>Within the context of training and testing a machine learning model, we can use Composition in Differentially Privacy and Privacy Amplication Theorem to derive the privacy loss in situations where we are iteritvely querying a model with a random subset of the data, such as in the case of training and testing Neural Networks. The summation of privacy losses over queries follows the intuition from the Fundamental Law of Information Recovery that continuously querying a private mechanism will increase the privacy loss incurred.</p>
<h2 id="differentially-private-stochastic-gradient-descent-dp-sgd">Differentially Private Stochastic Gradient Descent (DP-SGD)</h2>
<h3 id="introduction-to-dp-sgd">Introduction to DP-SGD</h3>
<p>The seminal paper that applies Differential Privacy to Deep Learning is <a href="https://arxiv.org/abs/1607.00133">Deep Learing with Differential Privacy</a>. In this paper, a differentially-private variant of stochastic gradient descent is proposed (DP-SGD).</p>
<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/138136971-c51261d3-73de-4afb-97b9-71e149ddfacd.png">
</p>
<center>
<a href="https://secml.github.io/class4">Source</a>  
</center>
<h3 id="estimating-differential-privacy-for-a-single-gradient-update">Estimating Differential Privacy for a Single Gradient Update</h3>
<p>As in vanilla SGD, we start with calculating the gradient of a batch of data. In order to limit the amount information we learn from the batch, we clip the gradient at C which is a hyperpareter for the algorithim. We than add noise <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><msup><mi>σ</mi><mn>2</mn></msup></mrow><annotation encoding="application/x-tex">\sigma^2</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.814108em; vertical-align: 0em;"></span><span class="mord"><span class="mord mathnormal" style="margin-right: 0.03588em;">σ</span><span class="msupsub"><span class="vlist-t"><span class="vlist-r"><span class="vlist" style="height: 0.814108em;"><span class="" style="top: -3.063em; margin-right: 0.05em;"><span class="pstrut" style="height: 2.7em;"></span><span class="sizing reset-size6 size3 mtight"><span class="mord mtight">2</span></span></span></span></span></span></span></span></span></span></span></span> proportional C and than update the parameters of the model.</p>
<h3 id="estimating-differential-privacy-for-accross-gradient-updates">Estimating Differential Privacy for accross Gradient Updates</h3>
<p>In order to get the total privacy loss, we must aggregate the privacy loss for each gradient update. Based on Composition in Differential Privacy and The Privacy Amplicaiton Theorem, we can define a Naive estimate of the upper bound of the privacy loss. Given an (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private algorithim, with a batch size proportional to <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>q</mi></mrow><annotation encoding="application/x-tex">q</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.625em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span></span></span></span></span> run for T iterations has a differential privacy of (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>q</mi><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">Tq\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.87777em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">Tq</span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>q</mi><mi>δ</mi></mrow><annotation encoding="application/x-tex">Tq\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">Tq</span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>).</p>
<p>It turns out that a more complex analysis can yield lower bounds for the privacy loss. Thus, finding the lowest bound possible allows us to more accurately determine the privacy loss and to avoid overstimating it. For example, the Strong Compositon Theorem can be used to prove an even lower bound on the privacy loss. In the DP-SGD paper, the authors propose the Moments Accountant which provided the lowest bound</p>
<ul>
<li><strong>Naive Analysis:</strong> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>q</mi><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">Tq\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.87777em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">Tq</span><span class="mord mathnormal">ϵ</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>q</mi><mi>δ</mi></mrow><annotation encoding="application/x-tex">Tq\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">Tq</span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private</li>
<li><strong>Strong Composition Theorem:</strong> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><mi>q</mi><mi>ϵ</mi><msqrt><mrow><mi>T</mi><mi>l</mi><mi>o</mi><mi>g</mi><mo stretchy="false">(</mo><mn>1</mn><mi mathvariant="normal">/</mi><mi>δ</mi><mo stretchy="false">)</mo></mrow></msqrt><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(q\epsilon\sqrt{Tlog(1/\delta)})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.24em; vertical-align: -0.305em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span><span class="mord mathnormal">ϵ</span><span class="mord sqrt"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.935em;"><span class="svg-align" style="top: -3.2em;"><span class="pstrut" style="height: 3.2em;"></span><span class="mord" style="padding-left: 1em;"><span class="mord mathnormal" style="margin-right: 0.01968em;">Tl</span><span class="mord mathnormal">o</span><span class="mord mathnormal" style="margin-right: 0.03588em;">g</span><span class="mopen">(</span><span class="mord">1/</span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span><span class="mclose">)</span></span></span><span class="" style="top: -2.895em;"><span class="pstrut" style="height: 3.2em;"></span><span class="hide-tail" style="min-width: 1.02em; height: 1.28em;"><svg width="400em" height="1.28em" viewBox="0 0 400000 1296" preserveAspectRatio="xMinYMin slice"><path d="M263,681c0.7,0,18,39.7,52,119
c34,79.3,68.167,158.7,102.5,238c34.3,79.3,51.8,119.3,52.5,120
c340,-704.7,510.7,-1060.3,512,-1067
l0 -0
c4.7,-7.3,11,-11,19,-11
H40000v40H1012.3
s-271.3,567,-271.3,567c-38.7,80.7,-84,175,-136,283c-52,108,-89.167,185.3,-111.5,232
c-22.3,46.7,-33.8,70.3,-34.5,71c-4.7,4.7,-12.3,7,-23,7s-12,-1,-12,-1
s-109,-253,-109,-253c-72.7,-168,-109.3,-252,-110,-252c-10.7,8,-22,16.7,-34,26
c-22,17.3,-33.3,26,-34,26s-26,-26,-26,-26s76,-59,76,-59s76,-60,76,-60z
M1001 80h400000v40h-400000z"></path></svg></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.305em;"><span class=""></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>T</mi><mi>q</mi><mi>δ</mi></mrow><annotation encoding="application/x-tex">Tq\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.88888em; vertical-align: -0.19444em;"></span><span class="mord mathnormal" style="margin-right: 0.03588em;">Tq</span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private</li>
<li><strong>Moments Accountant:</strong> (<span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><mi>q</mi><mi>ϵ</mi><msqrt><mi>T</mi></msqrt><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(q\epsilon\sqrt{T})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.17667em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span><span class="mord mathnormal">ϵ</span><span class="mord sqrt"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.926665em;"><span class="svg-align" style="top: -3em;"><span class="pstrut" style="height: 3em;"></span><span class="mord" style="padding-left: 0.833em;"><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span></span></span><span class="" style="top: -2.88666em;"><span class="pstrut" style="height: 3em;"></span><span class="hide-tail" style="min-width: 0.853em; height: 1.08em;"><svg width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z"></path></svg></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.113335em;"><span class=""></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>)-differentially private.</li>
</ul>
<h3 id="moments-accountant">Moments Accountant</h3>
<p>The fundamental insight of the Moments Accountant technique is that the privacy loss is a random variable. Thus, if we look at the distribution of the privacy loss, we can see that <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>ϵ</mi></mrow><annotation encoding="application/x-tex">\epsilon</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.43056em; vertical-align: 0em;"></span><span class="mord mathnormal">ϵ</span></span></span></span></span> defines the privacy budget for the loss and <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span> provides an upper bound on the tail of the distribution.</p>
<p align="center">
<img width="500" alt="Screen Shot 2021-10-20 at 1 50 02 PM" src="https://user-images.githubusercontent.com/34798787/138145128-0d67885f-a5c4-448f-9fcd-78fa0452a3ce.png">
</p>
<center>
<a href="https://www.youtube.com/watch?v=jm1Sfdno_5A">Source</a>  
</center>
<p>By treating the privacy loss as a random variable, we can leverage probability theory to derive a lower bound on the privacy loss. Specifically, the Moments Accountant Technique uses the moments of the distribution (ie mean, variance, skewness and kurtosis) in order to do so. Incorporating the higher order information of the distibution made available through the moments  allows us to derive a lower bound on the privacy loss: <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>O</mi><mo stretchy="false">(</mo><mi>q</mi><mi>ϵ</mi><msqrt><mi>T</mi></msqrt><mo stretchy="false">)</mo></mrow><annotation encoding="application/x-tex">O(q\epsilon\sqrt{T})</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 1.17667em; vertical-align: -0.25em;"></span><span class="mord mathnormal" style="margin-right: 0.02778em;">O</span><span class="mopen">(</span><span class="mord mathnormal" style="margin-right: 0.03588em;">q</span><span class="mord mathnormal">ϵ</span><span class="mord sqrt"><span class="vlist-t vlist-t2"><span class="vlist-r"><span class="vlist" style="height: 0.926665em;"><span class="svg-align" style="top: -3em;"><span class="pstrut" style="height: 3em;"></span><span class="mord" style="padding-left: 0.833em;"><span class="mord mathnormal" style="margin-right: 0.13889em;">T</span></span></span><span class="" style="top: -2.88666em;"><span class="pstrut" style="height: 3em;"></span><span class="hide-tail" style="min-width: 0.853em; height: 1.08em;"><svg width="400em" height="1.08em" viewBox="0 0 400000 1080" preserveAspectRatio="xMinYMin slice"><path d="M95,702
c-2.7,0,-7.17,-2.7,-13.5,-8c-5.8,-5.3,-9.5,-10,-9.5,-14
c0,-2,0.3,-3.3,1,-4c1.3,-2.7,23.83,-20.7,67.5,-54
c44.2,-33.3,65.8,-50.3,66.5,-51c1.3,-1.3,3,-2,5,-2c4.7,0,8.7,3.3,12,10
s173,378,173,378c0.7,0,35.3,-71,104,-213c68.7,-142,137.5,-285,206.5,-429
c69,-144,104.5,-217.7,106.5,-221
l0 -0
c5.3,-9.3,12,-14,20,-14
H400000v40H845.2724
s-225.272,467,-225.272,467s-235,486,-235,486c-2.7,4.7,-9,7,-19,7
c-6,0,-10,-1,-12,-3s-194,-422,-194,-422s-65,47,-65,47z
M834 80h400000v40h-400000z"></path></svg></span></span></span><span class="vlist-s">​</span></span><span class="vlist-r"><span class="vlist" style="height: 0.113335em;"><span class=""></span></span></span></span></span><span class="mclose">)</span></span></span></span></span>, <span class="katex--inline"><span class="katex"><span class="katex-mathml"><math xmlns="http://www.w3.org/1998/Math/MathML"><semantics><mrow><mi>δ</mi></mrow><annotation encoding="application/x-tex">\delta</annotation></semantics></math></span><span class="katex-html" aria-hidden="true"><span class="base"><span class="strut" style="height: 0.69444em; vertical-align: 0em;"></span><span class="mord mathnormal" style="margin-right: 0.03785em;">δ</span></span></span></span></span>). For a deeper look into the Moments Accountant technique, refer to <a href="https://arxiv.org/abs/1607.00133">Deep Learing with Differential Privacy</a>.</p>
<h2 id="private-aggregation-of-teacher-ensembles-pate">Private Aggregation of Teacher Ensembles (PATE)</h2>
<h3 id="introduction-to-pate">Introduction to PATE</h3>
<p>In <a href="https://arxiv.org/abs/1802.08908">Scalable Private Learning with PATE</a>, authors set out solve the problem of preserving the privacy of training data when training classfiers. They began with defining certain criteria for the solution:</p>
<ul>
<li>Differential privacy protection guarentees</li>
<li>Intuitive privacy protection guarentees</li>
<li><strong>Independent of learning algorithim</strong></li>
</ul>
<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/139095511-ff27899c-80f0-45bf-a4d0-d18820c20e34.png">
</p>
<center>
<a href="https://www.youtube.com/watch?v=cjo_u_yT2wQ&amp;t=1s">Source</a>  
</center>
<h3 id="ensembling">Ensembling</h3>
<p>In differential privacy, we seek to learn general trends from the data. That is, the outcome of the the prediction of a sample, should be the same whether or not we choose to include an sample in the dataset for each sample in the dataset. One natural way to achieve this to use an ensemble of models trained on random subsets of the dataset. In this way, the prediction for a specific sample is less likely to depend on a single example. Furthermore, if there is a strong consensus among the ensemble, it is likely the prediction stems from a general trend rather than a specific sample. In practice, this allows us to define a lower, data dependent bound on the privacy loss.</p>
<p>Although ensembling enhances differentially privacy, it is still possible that the inclusion/exclusion of samples in the various teachers will change the outcome of the predictions. This is especially the case when there is not a strong consensus among the ensemble; the vote of single teacher can sway the prediction of the output.</p>
<h3 id="noisy-aggregation">Noisy Aggregation</h3>
<p>In order to address the aforementioned shortcoming, we can aggregate the votes of the teachers in a noisy way. This is realized by adding noise to the final prediction.</p>
<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/139089966-fa783578-cc52-4a58-bcb4-9a06432868ac.png">
</p>
<center>
<a href="https://www.youtube.com/watch?v=cjo_u_yT2wQ&amp;t=1s">Source</a>  
</center>
<h3 id="student-training">Student Training</h3>
<p>Although the aggregated teacher is a good step towards differential privacy, it does have some shortcomings:</p>
<ul>
<li>Each prediction increases total privacy loss. Thus, the privacy budget creates a tradeoff between the accuracy and number of predictions.</li>
<li>Inspection of internal may reveal private data. However, we want privacy guarentees that hold for white box adversaries.</li>
</ul>
<p>As a result, the PATE framework introduces a student network that is trained on publicly available data using labels from Teacher Model.</p>
<p align="center">
<img width="500" alt="dpsgd" src="https://user-images.githubusercontent.com/34798787/138165916-a61f044d-4d45-4f5f-8e62-3fca94b65b84.png">
</p>
<center>
<a href="https://www.youtube.com/watch?v=cjo_u_yT2wQ&amp;t=1s">Source</a>  
</center>
<h1 id="demo">Demo</h1>
<h2 id="overview">Overview</h2>
<p>The following demo consists of three sections:</p>
<ul>
<li><strong>Data Preparation</strong>
<ul>
<li>Define Dataset</li>
<li>Define Dataloader</li>
</ul>
</li>
<li><strong>Model Preparation</strong>
<ul>
<li>Model Defintion</li>
</ul>
</li>
<li><strong>Training and Validation</strong>
<ul>
<li>Training Teacher Ensemble</li>
<li>Generating Labelled Public Dataset</li>
<li>Train and Evaluate Student Model</li>
</ul>
</li>
<li><strong>Interpret Results</strong>
<ul>
<li>Accuarcy</li>
<li>Privacy Loss</li>
</ul>
</li>
</ul>
</div>
</body>

</html>






