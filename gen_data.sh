#baseline
#sh gen.sh comp550/ sentiment_sst_binary
#sh gen.sh comp550/ sentiment_imdb
#sh gen.sh comp550/ amzsst_raw
#language model
#sh gen.sh comp550/ amzlm &
#sh gen.sh comp550/ amzlm_book &
#sh gen.sh comp550/ amzlm_movie &
#sh gen.sh comp550/ amzlm_elec &
#imdb classifier
#sh gen.sh comp550/ amzcls
#sh gen.sh comp550/ amzcls_book
#sh gen.sh comp550/ amzcls_movie
#sh gen.sh comp550/ amzcls_elec

#sst classifier
sh gen.sh comp550/ amzsst &
sh gen.sh comp550/ amzsst_book &
sh gen.sh comp550/ amzsst_movie &
sh gen.sh comp550/ amzsst_elec &
