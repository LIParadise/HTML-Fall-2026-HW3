#set text(size: 13pt, font: ((name: "Libertinus Serif", covers: "latin-in-cjk"), "Noto Serif CJK JP"))

#show math.sum: math.limits
#show math.product: math.limits
#show math.xor: math.limits
#show math.xor.big: math.limits
#show math.union: math.limits
#show math.union.big: math.limits
#show math.inter: math.limits
#show math.inter.big: math.limits
#show math.integral: math.limits
#show link: set text(fill: blue)
#set quote(block: true)

#import "@preview/commute:0.3.0": arr, commutative-diagram, node

#let skip_vector_marker = (
  sym.dots,
  sym.dots.h,
  sym.dots.h.c,
  sym.dots.v,
  sym.dots.up,
  sym.dots.down,
)
#let hl(content) = box(
  content,
  inset: 0.1em,
  radius: 0.2em,
  fill: gradient.linear(
    rgb("#F5AAAA"), // Soft coral red
    rgb("#FFD0A6"), // Peach orange
    rgb("#FFF0B3"), // Warm pastel yellow
    angle: 33deg,
  ),
)
#let quo(expr) = {
  set math.frac(style: "skewed")
  expr
}
#let row(expr) = {
  $cancel(#h(1em) expr #h(1em), angle: #90deg, stroke: #(dash: "dashed", paint: black.transparentize(67%)))$
}
#let col(expr) = {
  $cancel(#v(1em) expr #v(1em), angle: #0deg, stroke: #(dash: "dashed", paint: black.transparentize(67%)))$
}
#let rows(..exprs) = {
  let arr = exprs.pos().map(expr => if expr in skip_vector_marker [ #expr ] else [#row(expr)])
  math.vec(..arr)
}
#let cols(..exprs) = {
  let arr = exprs.pos().map(expr => if expr in skip_vector_marker [ #expr ] else [#col(expr)])
  math.mat(..arr)
}

#let aut = $op("Aut")$
#let endo = $op("End")$
#let ann = $op("Ann")$
#let hom = $op("Hom")$
#let subgroup = $subset.eq^"group"$
#let subideal = $subset.eq^"ideal"$
#let submodule = $subset.eq^"module"$
#let subring = $subset.eq^"ring"$
#let inj = $arrow.r.hook$
#let surj = $arrow.r.twohead$
#let iso = $arrow.r.tilde$
#let where = $mid(|)$
#let diag = $op("diag")$

#let ind = $bb(1)$
#let lr = $op("LR")$
#let llr = $op("LLR")$
#let lrt = $"LRT"$
#let pr = $op("Pr")$
#let given = $mid(|)$
#let vs = $mid(bar.v.double)$

= Homework 3

== Linear Regression

1. Linear regression when $(bold(x), y) in RR times {plus.minus 1}$ with square error
  - *B*: $(sum_(i=1)^n y_i x_i)/(sum_(i=1)^n x_i^2)$
  - We've derived during in class that for square error w.r.t. linear hypotheses, $ bold(w) = (X^T X)^(-1) X bold(y) $ by observing the parabolic convex shape of the error, where $X in M_(n times d) (RR)$ (not explicitly writing down the offset dimension here)
  - Plugging in $d = 1$ we get the $(X^T X)^(-1)$ part as $1 / (sum_(i=1)^n x_n^2)$
  - Similarly $X bold(y) = sum_(i=1)^n x_i y_i$

== Logistic Regression

2. Choose the _false_ one from the listed claims
  - *C*: Mean as MLE for Laplace: turns out Laplace MLE is sample median instead of sample mean
  - Observation/interpretation: is it true that for random processes i.i.d. following some distribution, that the mean must be the maximum likelihood estimator?
    - Some distributions don't even have either mean or variance: #link("https://stats.stackexchange.com/questions/36027")[Cauchy]
  - Turns out such suspicion is indeed justified: even if the moments exist, they may not be the maximum likelihood estimator, since in the case of Laplace distribution it's indeed the #link("https://math.stackexchange.com/a/240503/595630")[sample median] that's the maximum likelihood estimator

3. Likelihood estimator for uniform $[0, theta]$ when given samples ${y_i where 1<=i<=n}$ in the case of $hat(theta) > limits(max)_i {y_i}$
  - *A*: $1 / hat(theta)^n$
  - Takeaway: maximum likelihood estimator may actually be biased!
    - The German tank #link("https://mathworld.wolfram.com/GermanTankProblem.html")[problem], as illustrated also by #link("https://www.youtube.com/watch?v=WLCwMRJBhuI")[Numberphile]
    - MLE is simply taking the max of the sample, but it in general _differs_ from the unbiased estimator: when taken expected value (i.e. the ensemble average) is actually the latter, in addition the latter has a smaller variance
    - But this example is uniform _discrete_ and drawn _without replacement_...
  - In any case since we've assumed uniform distribution i.i.d. (discrete) from $op("Unif")([0, theta])$, the likelihood is simply, assuming $hat(theta) in ZZ^+$, $1 / (hat(theta) + 1)^n$
  - I'll assume there's a typo...

4. What's the error function of the following PLA variant as given by $ bold(w)_(t+1) <- bold(w)_t + eta/N sum_(y_n != op("sgn")(bold(w)_n^T bold(x)_n)) y_n bold(x)_n $
  - *B*: $op("err")(bold(w), bold(x), y) = max(0, -y bold(w)^T bold(x))$
  - $eta$ could be interpreted as the learning rate i.e. the term we multiply on the gradient
  - $1/N$ could be interpreted as averaging over pointwise error term
  - Meaning the pointwise error should be constant for correct terms and $-y bold(x)$ for error terms
  - $max(0, -y bold(w)^T bold(x))$ satisfies our observation: when correct, i.e. $y = op("sgn")(bold(w)^T bold(x))$, the error is always $0$ and thus gradient is $0$, while OTOH (exactly) when the sign differs, the gradient w.r.t. $bold(w)$ is $bold(x) y$
  - We rule out $max(0, 1-y bold(w)^T bold(x))$ since with this error function, all those $abs(bold(w)^T bold(x)) < 1$ terms shall contribute to the (batched) gradient descent step, but in our case it's instead exactly the set of points with $y != op("sgn")(bold(w)^T bold(x))$ that contributes nonzero amount

5. Calculate gradient of pointwise exponential error $ op("err")(bold(w), bold(x), y) := e^(-y bold(x)^T bold(w)) $
  - *A*: $y dot e^(-y bold(x)^T bold(w)) dot bold(x)$
  - $
      partial / (partial w_i) op("err")(bold(w), bold(x), y) &=^"defn" partial / (partial w_i)e^(-y bold(x)^T bold(w)) \
      &=^"chain" e^(-y bold(x)^T bold(w)) times partial / (partial w_i) (-y bold(x)^T bold(w)) \
      &=^"bilinear" e^(-y bold(x)^T bold(w)) times (-y x_i)
    $
  - Collect all indices and write in column vector, we arrive at $ nabla_bold(w) op("err")(bold(w), bold(x), y) = e^(-y bold(x)^T bold(w)) times (-y bold(x)) $
  - Gradient descent goes _negative_ direction so the update term would be (setting unit learning rate) $ y dot e^(-y bold(x)^T bold(w)) dot bold(x) $

6. Arrange $K$ linear regression hypotheses as columns in a matrix $W = cols(bold(w)_1, dots, bold(w)_K)$ and classify $bold(x)$ based on $ limits(arg max)_i mat(q_1, dots, q_K) $ where $ bold(q) := (q_1, dots, q_K) := scr(n)(sigma(bold(x) W)) $ where $sigma$ applies logistic/sigmoid to each entry, $scr(n)$ normalizes to probability vector s.t. $scr(n)(sigma(bold(x) W))$ is a well-defined conditional probability (row) vector. Denote also $q_y$ as $h_y (bold(x))$ to signify the dependence on input $bold(x)$ and the conditional probability as given by the $y$-th hypothesis. The error function w.r.t. to the $y$-th hypothesis $bold(w)_y$ is found to be $ op("err")(W, bold(x), y) = -ln h_y (bold(x)) $ Calculate the partial derivative w.r.t. its $i$-th entry.
  - *B*: $(h_k (bold(x)) - ind(y=k)) x_i$
  - Plugging in the terms we solve for the $y$-th hypothesis, of which partial derivative w.r.t. $i$-th entry i.e. $W_(i y)$
    $
      partial / (partial W_(i y)) (- ln h_y (bold(x))) & =^"defn" partial / (partial W_(i y)) -ln (e^(bold(w)_y^T bold(x)) / (sum_(i=1)^K e^(bold(w)_i^T bold(x))) ) \
      &=^ln partial / (partial W_(i y)) ln (sum_(i=1)^K e^(bold(w)_i^T bold(x))) - partial / (partial W_(i y)) ln(e^(bold(w)_y^T bold(x))) \
      &=^(ln e^x = x) partial / (partial W_(i y)) ln (sum_(i=1)^K e^(bold(w)_i^T bold(x))) - partial / (partial W_(i y)) bold(w)_y^T bold(x) \
      &=^"chain" (e^(bold(w)_y^T bold(x)) dot x_i) / (sum_(i=1)^K e^(bold(w)_i^T bold(x))) - x_i \
      &=^"defn" x_i (h_y (bold(x)) - 1)
    $
  - This above is partial derivative w.r.t. the $i$-th entry of the $y$-th hypothesis, and one sees for other $k != y$-th hypothesis, partial differentiating $h_y (bold(x))$'s numerator, $e^(bold(w)_k^T bold(x))$, is zero, thus the indicator term.

== Linear Models for Classification

7. Assume binary classification takes linear time with constant $a$, what's time required for OVO training where there are $K$ classes each of which having $n/K$ sized data?
  - *B*: $a dot (K-1) dot n$
  - Per OVO instance would be $2 dot n/K$ size, which takes $2 dot a dot n/K$ time by premise
  - OVO needs $binom(K, 2)$ instances
  - $binom(K, 2) dot 2 dot a dot n/K = (K (K-1))/2 dot 2 dot a dot n/K = a dot (K-1) dot n$
