from multi_rake import Rake
from collections import Counter
import re

rake = Rake()
abstract_str="The microtubule-associated protein tau (MAPT) plays an important role in Alzheimer's disease and primary tauopathy diseases. The abnormal accumulation of tau contributes to the development of neurotoxicity, inflammation, neurodegeneration, and cognitive deficits in tauopathy diseases. Tau synergically interacts with amyloid-beta in Alzheimer's disease leading to detrimental consequence. Thus, tau has been an important target for therapeutics development for Alzheimer's disease and primary tauopathy diseases. Tauopathy animal models recapitulating the tauopathy such as transgenic, knock-in mouse and rat models have been developed and greatly facilitated the understanding of disease mechanisms. The advance in PET and imaging tracers have enabled non-invasive detection of the accumulation and spread of tau, the associated microglia activation, metabolic, and neurotransmitter receptor alterations in disease animal models. In vivo microPET studies on mouse or rat models of tauopathy have provided significant insights into the phenotypes and time course of pathophysiology of these models and allowed the monitoring of treatment targeting at tau. In this study, we discuss the utilities of PET and recently developed tracers for evaluating the pathophysiology in tauopathy animal models. We point out the outstanding challenges and propose future outlook in visualizing tau-related pathophysiological changes in brain of tauopathy disease animal models."
keywords = rake.apply(abstract_str)
top20_keywords=keywords
# 提取关键词列表中的词语
keywords_list = [kw[0] for kw in top20_keywords]
# 用正则表达式找到每个关键词在原文中出现的次数
word_frequencies = Counter()
for keyword in keywords_list:
    # 使用正则匹配关键词的出现次数，忽略大小写
    count = len(re.findall(re.escape(keyword), abstract_str))
    word_frequencies[keyword] = count
print(word_frequencies)