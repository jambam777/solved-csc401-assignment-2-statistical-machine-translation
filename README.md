Download Link: https://assignmentchef.com/product/solved-csc401-assignment-2-statistical-machine-translation
<br>
This assignment will give you experience in working with <em>n</em>-gram models, smoothing, and statistical machine translation through word alignment. Knowledge of French is not required.

Your tasks are to build bigram and unigram models of English and French, to smooth their probabilities using add-<em>δ </em>discounting, to build a world-alignment model between English and French using the IBM-1 model, and to use these models to translate French sentences into English with a decoder that we provide. The programming language for this assignment is Python3.

<h1>2      Background</h1>

<strong>Canadian Hansard data</strong>

The main corpus for this assignment comes from the official records (<em>Hansards</em>) of the 36<em><sup>th </sup></em>Canadian Parliament, including debates from both the House of Representatives and the Senate. This corpus is available at /u/cs401/A2 SMT/data/Hansard/ and has been split into Training/ and Testing/ directories.

This data set consists of pairs of corresponding files (*.e is the English equivalent of the French *.f) in which every line is a sentence. Here, sentence alignment has already been performed for you. That is, the <em>n<sup>th </sup></em>sentence in one file corresponds to the <em>n<sup>th </sup></em>sentence in its corresponding file (e.g., line <em>n </em>in fubar.e is aligned with line <em>n </em>in fubar.f). Note that this data only consists of sentence pairs; many-to-one, many-to-many, and one-to-many alignments are not included.

Furthermore, for the purposes of this assignment we have filtered this corpus down to sentences with between approximately 4 and 15 tokens to simplify the computational requirements of alignment and decoding. We have also converted the file encodings from ISO-8859-1 to ASCII so as to further simplify the problem. This involved <em>transliterating </em>the original text to remove <em>diacritics</em>, i.e., accented characters (e.g., <em>Chr´etien </em>becomes <em>Chretien</em>).

To test your code, you may like to use the samples provided at /u/cs401/A2 SMT/data/Toy/.

<h2>Add-<em>δ </em>smoothing</h2>

Recall that the maximum likelihood estimate of the probability of the current word <em>w<sub>t </sub></em>given the previous word <em>w<sub>t</sub></em>−<sub>1 </sub>is

<em>.                                                                        </em>(1)

Copyright     2019 Mohamed Abdalla, Frank Rudzicz. All rights reserved.

<em>Count</em>(<em>w<sub>t</sub></em><sub>−1</sub><em>,w<sub>t</sub></em>) refers to the number of times the word sequence <em>w<sub>t</sub></em><sub>−1</sub><em>w<sub>t </sub></em>appears in a training corpus, and <em>Count</em>(<em>w<sub>t</sub></em><sub>−1</sub>) refers to the number of times the word <em>w<sub>t</sub></em><sub>−1 </sub>appears in that corpus.

Laplace’s method of add-1 smoothing for <em>n</em>-grams simulates observing otherwise unseen events by providing probability mass to those unseen events by discounting events we have seen. Although the simplest of all smoothing methods, in practice this approach does not work well because too much of the <em>n</em>-gram probability mass is assigned to unseen events, thereby increasing the overall entropy unacceptably.

Add-<em>δ </em>smoothing generalizes Laplace smoothing by adding <em>δ </em>to the count of each bigram, where 0 <em>&lt; δ </em>≤ 1, and normalizing accordingly. This method is generally attributed to G.J. Lidstone<a href="#_ftn1" name="_ftnref1"><sup>[1]</sup></a>. Given a known vocabulary V of size kVk, the probability of the current word <em>w<sub>t </sub></em>given the previous word <em>w<sub>t</sub></em><sub>−1 </sub>in this model is

<h2>1. Preprocess input text</h2>

First, implement the following Python function:

preprocess(in sentence, language)

that returns a version of the input sentence in sentence that is more amenable to training. For both languages, separate sentence-final punctuation (sentences have already been determined for you), commas, colons and semicolons, parentheses, dashes between parentheses, mathematical operators (e.g., +, -, <em>&lt;</em>, <em>&gt;</em>, =), and quotation marks. Certain contractions are required in French, often to eliminate vowel clusters. When the input language is ‘french’, separate the following contractions:

<table width="0">

 <tbody>

  <tr>

   <td width="213"><strong>Type</strong></td>

   <td width="205"><strong>Modification</strong></td>

   <td width="197"><strong>Example</strong></td>

  </tr>

  <tr>

   <td width="213">Singular definite article</td>

   <td width="205">Separate leading <strong>l’ </strong>from</td>

   <td width="197"><em>l’election </em>⇒ <em>l’ election</em></td>

  </tr>

  <tr>

   <td width="213">(<em>le</em>, <em>la</em>)</td>

   <td width="205">concatenated word</td>

   <td width="197"> </td>

  </tr>

  <tr>

   <td width="213">Single-consonant words</td>

   <td width="205">Separate leading consonant</td>

   <td width="197"><em>je t’aime </em>⇒ <em>je t’ aime</em>,</td>

  </tr>

  <tr>

   <td width="213">ending        in      e-‘muet’         (e.g.,‘dropped’-e <em>ce</em>, <em>je</em>)</td>

   <td width="205">and apostrophe from concatenated word</td>

   <td width="197"><em>j’ai </em>⇒ <em>j’ ai</em></td>

  </tr>

  <tr>

   <td width="213"><em>que</em></td>

   <td width="205">Separate leading <strong>qu’ </strong>from</td>

   <td width="197"><em>qu’on </em>⇒ <em>qu’ on</em>,</td>

  </tr>

  <tr>

   <td width="213"> </td>

   <td width="205">concatenated word</td>

   <td width="197"><em>qu’il </em>⇒ <em>qu’ il</em></td>

  </tr>

  <tr>

   <td width="213">Conjunctions</td>

   <td width="205">Separate following <em>on </em>or <em>il</em></td>

   <td width="197"><em>puisqu’on </em>⇒ <em>puisqu’ on</em>,</td>

  </tr>

  <tr>

   <td width="213"><em>puisque </em>and <em>lorsque</em></td>

   <td width="205"> </td>

   <td width="197"><em>lorsqu’il </em>⇒ <em>lorsqu’ il</em></td>

  </tr>

 </tbody>

</table>

Any words containing apostrophes not encapsulated by the above rules can be left as-is. Additionally, the following French words should not be separated: <em>d’abord</em>, <em>d’accord</em>, <em>d’ailleurs</em>, <em>d’habitude</em>.

A template of this function has been provided for you at /u/cs401/A2 SMT/code/preprocess.py. Make your changes to a copy of this file and submit your version.

<h2>2. Compute <em>n</em>-gram counts</h2>

Next, implement a function to simply count all unigrams and bigrams in the preprocessed training data, namely:

LM = lm train(data dir, language, fn LM)

that returns a special language model structure (a dictionary), LM, defined below. This function trains on all of the data files in data dir that end in either ‘e’ for English or ‘f’ for French (which is specified in the argument language) and saves the structure that it returns in the filename fn LM.

The structure returned by this function should be called ‘LM’ and must have two keys: ‘uni’ and ‘bi’, each of which holds structures (additional dictionaries) which incorporate unigram and bigram counts, respectively. The fieldnames (i.e. <em>keys</em>) to the ‘uni’ structure are words and the values of those fields are the total counts of those words in the training data. The keys to the ‘bi’ structure are words (<em>w<sub>t</sub></em><sub>−1</sub>) and their values are dictionaries. The keys of those sub-dictionaries are also words (<em>w<sub>t</sub></em>) and the values of those fields are the total counts of ‘<em>w<sub>t</sub></em><sub>−1</sub><em>w<sub>t</sub></em>’ in the training data.

E.g.,

&gt;&gt; LM[‘uni’][‘word’] = 5 % the word ‘word’ appears 5 times in training

&gt;&gt; LM[‘bi’][‘word’][‘bird’] = 2 % the bigram ‘word bird’ appears twice in training

A template of this function has been provided for you at /u/cs401/A2 SMT/code/lm train.py. Note that this template calls preprocess.

Make your changes to a copy of the lm train.py template and submit your version. Train two language models, one for English and one for French, on the data at /u/cs401/A2 SMT/data/Hansard/Training/. You will use these models for subsequent tasks.

<h2>3. Compute log-likelihoods and add-<em>δ </em>log-likelihoods</h2>

Now implement a function to compute the log-likelihoods of test sentences, namely:

logProb = lm prob(sentence, LM, smoothing, delta, vocabSize) .

This function takes sentence (a previously preprocessed string) and a language model LM (as produced by lm train). If the argument smoothing is (‘False’), this function returns the maximum-likelihood estimate of the sentence. If the argument type is ‘True’, this function returns a <em>δ</em>-smoothed estimate of the sentence. In the case of smoothing, the arguments delta and vocabSize must also be specified (where 0 <em>&lt; δ </em>≤ 1).

When computing your MLE estimate, if you encounter the situation where 0, then assume that the probability <em>P</em>(<em>w<sub>t</sub></em><sub>+1 </sub>|<em>w<sub>t</sub></em>) = 0 or, equivalently, log<em>P</em>(<em>w<sub>t</sub></em><sub>+1 </sub>|<em>w<sub>t</sub></em>) = −∞. Negative infinity in Python is represented by float(‘-inf’). <strong>Use log base 2 (i.e. </strong>log2()<strong>).</strong>

A template of this function has been provided for you at /u/cs401/A2 SMT/code/log prob.py. Make your changes to a copy of the log prob.py template and submit your version.

We also provide you with the function /u/cs401/A2 SMT/code/perplexity.py, which returns the perplexity of a test corpus given a language model. You do not need to modify this function. Using the language models learned in Task 2, compute the perplexity of the data at /u/cs401/A2 SMT/data/Hansard/Testing/ for each language and for both the MLE and add-<em>δ </em>versions. Try at least 3 to 5 different values of <em>δ </em>according to your judgment. Submit a report, Task3.txt, which summarizes your findings. Your report can additionally include experiments on the log-probabilities of individual sentences.

<h2>4. Implement IBM-1</h2>

Now implement the IBM-1 algorithm to learn word alignments between English and French words, namely:

AM = align ibm1(train dir, num sentences, max iter, fn AM).

This function trains on the first num sentences read in data files from train dir. The parameter max iter specifies the maximum number of times the EM algorithm iterates before being terminated. This function returns a specialized alignment model structure, AM, in which AM[‘eng word’][‘fre word’] holds the probability (not log probability) of the word eng word aligning to fre word. In this sense, AM is essentially the <em>t </em>distribution from class, e.g.,

&gt;&gt; AM[‘bird’][‘oiseau’] = 0.8 % <em>t</em>(<em>oiseau</em>|<em>bird</em>) = 0<em>.</em>8

Here, we will use a simplified version of IBM-1 in which we ignore the NULL word and we ignore alignments where an English word would align with no French word, as discussed in class. So, the probability of an alignment <em>A </em>of a French sentence <em>F</em>, given a known English sentence <em>E </em>is

<em>len<sub>F</sub></em>

<em>P</em>(<em>A,F </em>|<em>E</em>) = <sup>Y </sup><em>t</em>(<em>f<sub>j </sub></em>|<em>e<sub>a</sub></em><em><sub>j</sub></em>)

<em>j</em>=1

where <em>a<sub>j </sub></em>is the index of the word in <em>E </em>which is aligned with the <em>j<sup>th </sup></em>word in <em>F </em>and <em>len<sub>F </sub></em>is the number of tokens in the French sentence. Since we are only using IBM-1, we employ the simplifying assumption that every alignment is equally likely.

<strong>Note</strong>: The na¨ıve approach to initializing AM is to have a uniform distribution over all possible English (e) and French (f) words, i.e., AM[‘e’][‘f’] = 1<em>/</em>kV<em><sub>F</sub></em>k, where kV<em><sub>F</sub></em>k is the size of the French vocabulary. Doing so, however, will consume too much memory and computation time. Instead, you can initialize AM[‘e’] uniformly over only those French words that occur in corresponding French sentences. For example,

<table width="0">

 <tbody>

  <tr>

   <td width="132"><em>the house</em></td>

   <td width="164"><em>la maison</em></td>

  </tr>

  <tr>

   <td width="132"><em>house of commons</em></td>

   <td width="164"><em>chambre des communes</em></td>

  </tr>

  <tr>

   <td width="132"><em>Andromeda galaxy</em></td>

   <td width="164"><em>galaxie d’Andromede</em></td>

  </tr>

 </tbody>

</table>

given only the training sentence pairs, you would initial-

ize the structure AM[‘house’][‘la’] = 0.2, AM[‘house’][‘maison’] = 0.2, AM[‘house’][‘chambre’] = 0.2, AM[‘house’][‘des’] = 0.2, AM[‘house’][‘communes’] = 0.2. There would be no probability of generating <em>galaxie </em>from <em>house</em>. <strong>Note </strong>that you can force AM[‘SENTSTART’][‘SENTSTART’] = 1 and

AM[‘SENTEND’][‘SENTEND’] = 1.

A template of this function has been provided for you at /u/cs401/A2 SMT/code/align ibm1.py. You will notice that we have suggested a general structure of empty helper functions here, but you are free to implement this function as you wish, as long as it meets with the specifications above. Make your changes to a copy of the align ibm1.py template and submit your version.

<h2>5. Translate and evaluate the test data</h2>

You will now produce your own translations, obtain reference translations from Google and the Hansards, and use the latter to evaluate the former, with a BLEU score. This will all be done in the file evalAlign.py (there is a very sparse template of this file at /u/cs401/A2 SMT/code/). To decode, we are providing the function english = decode(french, LM, AM),

at /u/cs401/A2 SMT/code/decode.py. Here, french is a preprocessed French sentence, LM and AM are your English language model from Task 2 and your alignment model trained from Task 4, respectively, and lmtype, delta, and vocabSize parameterize smoothing, as before in Task 3. You do not need to change the decode function, but you may (see the Bonus section, below).

For evaluation, translate the 25 French sentences in /u/cs401/A2 SMT/data/Hansard/Testing/Task5.f with the decode function and evaluate them using corresponding reference sentences, specifically:

<ol>

 <li>/u/cs401/A2SMT/data/Hansard/Testing/Task5.e, from the Hansards.</li>

 <li>/u/cs401/A2SMT/data/Hansard/Testing/Task5.google.e, Google’s translations of the French phrases<a href="#_ftn2" name="_ftnref2"><sup>[2]</sup></a>.</li>

</ol>

To evaluate each translation, use the BLEU score from lecture 6, i.e.,

<em>BLEU </em>= <em>BP<sub>C </sub></em>× (<em>p</em><sub>1</sub><em>p</em><sub>2 </sub><em>…p<sub>n</sub></em>)<sup>(1<em>/n</em>)                                                                                                                   </sup>(3)

Repeat this task with at least four alignment models (trained on 1<em>K</em>, 10<em>K</em>, 15<em>K</em>, and 30<em>K </em>sentences, respectively) and with three values of <em>n </em>in the BLEU score (i.e., <em>n </em>= 1<em>,</em>2<em>,</em>3). You should therefore have 25×4×3 BLEU scores in your evaluation. Write a short subjective analysis of how the different references differ from each other, and whether using more than 2 of them might be better (or worse).

In all cases, you can use the MLE language model (i.e., specify lmtype = ‘’). Optionally, you can try additional alignment models, smoothed language model with varying <em>δ</em>, or other test data from other files in /u/cs401/A2 SMT/data/Hansard/Testing/.

Submit your evaluation procedure, evalAlign.py, along with a report, Task5.txt, which summarizes your findings. If you make any changes to any other files, submit those files as well.

<h2>Bonus</h2>

We will give bonus marks for innovative work going substantially beyond the minimal requirements. Your overall mark for this assignment cannot exceed 100%.

You may decide to pursue any number of tasks of your own design related to this assignment, although you should consult with the instructor or the TA before embarking on such exploration. Certainly, the rest of the assignment takes higher priority. Some ideas:

<ul>

 <li>Try additional smoothing methods (e.g., Good-Turing, Knesser-Ney) and re-run the experiments in Task 3, above. Submit your code and an associated discussion.</li>

 <li>Implement the IBM-2 model of word-alignment, otherwise replicating Task 4 above. Ideally, translate the test data using this model and compute the error, as you did for Task 5. How does this model compare to IBM-1? Submit your code and an associated discussion.</li>

 <li>We have not considered the null word when performing alignments. Re-implement the IBM-1 alignment model to include null words and the possibility that no English word aligns with a French word (or vice versa). Submit your code and an associated discussion.</li>

 <li>Perform substantial data analysis of the error trends observed in each method you implement. This must go well beyond the basic discussion already included in the assignment. Submit a report.</li>

 <li>The decoder we use here is extremely simple and incomplete. You can write your own decoder that attempts to find ˆ<em>e </em>= argmax<em><sub>e </sub>P</em>(<em>e</em>|<em>f</em>) using a heuristic <em>A</em><sup>∗ </sup>search, for example. Alternatively, what happens if you weight the contributions of the alignment and the language model to the overall probability? Section 25.8 of the Jurafsky &amp; Martin textbook offers some ideas on how to improve the decoder. Submit your code and an associated discussion, comparing the decoded results to those performed with the default decoder.</li>

 <li>Read the sequence-to-sequence tutorial at <a href="https://www.tensorflow.org/tutorials/seq2seq">https://www.tensorflow.org/tutorials/seq2seq</a> and apply it to these data. Is the performance significantly better (or different) than IBM-1 on these data?</li>

 <li>The website <a href="https://www.allsides.com/unbiased-balanced-news">https://www.allsides.com/unbiased-balanced-news</a> curates news articles on particular events or stories according to their perceived political bias, using the spectrum used in Assignment 1 (less ‘alternative’ news). We sampled a considerable amount of these data; they are available at /s/course/csc401/A2/allSides (1.6GB) for you to examine. Unfortunately, since a right-leaning report on a story is not strictly a translation of a right-leaning report (or vice versa), the normal approach to sentence alignment (or SMT generally) does not apply; in our experiments, performance was unacceptably random. You, however, may be more fortunate…</li>

</ul>

<a href="#_ftnref1" name="_ftn1">[1]</a> Lidstone, G. J. (1920) Note on the general case of the Bayes-Laplace formula for inductive or <em>a priori </em>probabilities. <em>Transactions of the Faculty of Actuaries </em>8:182–192.

<a href="#_ftnref2" name="_ftn2">[2]</a> See <a href="https://developers.google.com/api-client-library/python/apis/translate/v2">https://developers.google.com/api-client-library/python/apis/translate/v2</a><a href="https://developers.google.com/api-client-library/python/apis/translate/v2">,</a> but be prepared to pay.