FROM anibali/pytorch

RUN pip install transformers 
RUN pip install peft 
RUN pip install scikit-learn 
RUN pip install datasets
RUN pip install sentencepiece
RUN pip install protobuf