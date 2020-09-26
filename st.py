import streamlit as st

st.title("Medium Article Generator")


starting_sentence = st.text_input(label='Enter Starting Sentence')
@st.cache
def generate():
    import gpt_2_simple as gpt2

    global sess1
    sess = gpt2.start_tf_sess()
    gpt2.load_gpt2(sess)

    global single_text
    single_text = gpt2.generate(sess, prefix=starting_sentence, top_k=40, length=100,
                                return_as_list=True)[0]
    return single_text


model_load_state = st.text("Loading Gpt2 model ... ")
generate()
model_load_state.text('Loading Gpt2 ... Done! Have Fun:)')





st.write(single_text)

#print(single_text)
#generated_text = gpt2.generate(sess, prefix=start_sentence, top_k=40, length=100)
#single_text = gpt2.generate(sess, prefix='natural language processing is really great if  ', top_k=40, length=100,
#                            return_as_list=True)[0]

#st.write(generate("natural language "))