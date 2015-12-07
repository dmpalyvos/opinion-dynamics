# Opinion-Dynamics

Diploma thesis at the National Technical University of Athens on Opinion Dynamics in social networks. 

## Contents
1. thesis-matlab: Code used in the thesis, written in matlab.
2. reaseach: base for python code used in later research. Saved here in order to preserve the history. Recent versions of the code please can be found [here](https://github.com/dmpalyvos/opinions-research).

## Summary
Opinions are important for both our personal lives and society as a whole. They are formed during a social learning process in which people use their social network to exchange useful information about the world. This dynamic process is described by several mathematical models that define how the opinions and the network behave over time. Those models can be separated into two distinct categories.

In the first category there are models which make the assumption that the structure of the society remains stable. Those models have been studied extensively and their behavior is well understood. We will begin with a short analysis of the DeGroot and Friedkin-Johnsen(Kleinberg) models so that the reader can understand the basics of the opinion dynamics process. We will continue by introducing a new model which we will call "Meeting a Friend". The purpose of this model is to try to simulate the Friedkin-Johnsen model with a limited amount of information, a situation which resembles reality more closely.

In the second category there are models that allow structural changes in the society. Those changes happen as a result of changing opinions and vice versa. Due to the fact that a mathematical analysis of these models is quite challenging, we will take an experimental approach. We will begin with the Asymmetric Coevolutionary Model in which only the weights of the society graph can change. After that we will have a look at the model proposed by Hegselmann and Krause which allows for a completely dynamic network. We will introduce a variation of the model by limiting the amount of available information. Finally we will focus our experiments on two variations of the K-Nearest Neighbors Model as it hasn't been studied as extensively as the rest.

For each model we will study whether it converges, the speed of convergence and the opinions at the equilibrium. All the models that we tested converge. The convergence times differ depending on the model and its parameters. Finally none of the models reach a consensus for sets of parameters that are close to reality.
