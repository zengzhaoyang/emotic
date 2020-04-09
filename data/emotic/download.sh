filename="a.zip"
file_id="0B7sjGeF4f3FYQUVlZ3ZOai1ieEU"
query=`curl -c ./cookie.txt -s -L "https://drive.google.com/uc?export=download&id=${file_id}" \
| perl -nE'say/uc-download-link.*? href="(.*?)\">/' \
| sed -e 's/amp;//g' | sed -n 2p`
url="https://drive.google.com$query"
curl -b ./cookie.txt -L -o ${filename} $url

wget http://sunai.uoc.edu/emotic/emotic_files/annotations.zip
