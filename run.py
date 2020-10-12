from fitbert import FitBert

fb = FitBertNew(mask_token="@placeholder", model_name="bert-base-cased", ensemble=True)

masked_string = "Why Bert, you're looking @placeholder today!"
options = ['buff', 'handsome', 'strong']

ranked_options = fb.rank(masked_string, options=options)
print(ranked_options)