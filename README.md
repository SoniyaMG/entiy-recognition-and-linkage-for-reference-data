# entiy-recognition-and-linkage-for-reference-data

This is a research project, which aims at linking author name mentions in a reference to the correspoding target authors in DBLP network.

**Docker Installation**
1) download and unzip the repository
2) cd to Reference2Auth_web_app
3) Build the docker image
   `docker build -t the_beatles_app .`
4) Run the docker image
    `docker run -it -p 5000:5000 the_beatles_app`

     Initially you will see the logs containing below line:
     > \* Running on http://127.0.0.1:5000/ (Press CTRL+C to quit)

     wait for a while until you see these lines :
     
     > Forwarding          https://314d06b61420.ngrok.io -> http://localhost:5000 <br>
     > \* Running on http://314d06b61420.ngrok.io
5) copy the link (containing ngrok.io) and access the same in the browser

