# messing-with-ai
These were my summer experiments at AI training using the Hugging Face libraries. My goal was to create a AI model capable of identifying identical prepositions using the Hugging Face framework. I was working from the Hugging Face NLP course which is found at this link(https://huggingface.co/learn/nlp-course/chapter1/2?fw=pt). I didn't know how to properly use github, so these files are a little chaotically organised. I uploaded four batches of files before I moved on to working on Dijkstra's algorithm. With the hardware that I had to, AI work was really too high level for me to feel satisfied with. Either everything was extremely easy or the problems were in graphic card drivers. A basic overview of what I accomplished can be found on this page, further information can be seen in commit info . 

I started by training a GPT-2 Byte Pair Encoding tokenizer on two different datasets and just observing how the differences in training sets affected how the tokenizers handled text. My two different datasets were a database of film reviews and a database of wikipedia articles. The most significant difference is how the word 'addicted' was split. The wikipedia trained tokenizer split it into 'add' and 'icted' whereas the film review split it into 'addict' and 'ed'. This accurately showed the difference in content between the two datasets. 

I then moved onto to training bert models to detect symonyms. This was surprisingly complex just because of the difficulties in defining a symonym. My first attempt was simply from following the hugging face tutorial in chapter 3. After that, I used ChatGPT to make a list of symonyms and used to do a second training run, which was not very effective as the second training run resulted in the model 'unlearning' what it had learned the first time round. I then tried combining the datasets so only one training run was needed, which improved results. After that, I wrote a very short python program to test differnet models so results could be more effectively compared. 

