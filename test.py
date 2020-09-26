import gpt_2_simple as gpt2
import os
import requests



model_name = "124M"
if not os.path.isdir(os.path.join("models", model_name)):
	print(f"Downloading {model_name} model...")
	gpt2.download_gpt2(model_name=model_name)
check_point = 'run3_again'

sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)
gpt2.generate(sess,prefix='natural language',top_k=40,length=100)
single_text = gpt2.generate(sess,prefix='natural language processing is really great if  ',top_k=40,length=100, return_as_list=True)[0]
print(single_text)

