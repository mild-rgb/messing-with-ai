from tokenizers import Tokenizer 
sst_tokenizer = Tokenizer.from_file("test-bpe-tokenizer-sst2.json")
wikitext_tokenizer = Tokenizer.from_file("test-bpe-tokenizer-wikitext.json")

while(True):
    phrase = input('type your phrase here\n')
    sst_enc = sst_tokenizer.encode(phrase)
    wiki_enc = wikitext_tokenizer.encode(phrase)
    print(sst_enc.tokens)
    print('\n')
    print(wiki_enc.tokens)
    print('\n')
   
    
