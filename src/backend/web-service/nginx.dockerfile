FROM nginx:stable-alpine

ADD ./nginx/default.conf /etc/nginx/conf.d/default.conf
#instead of mapping the config file in the docker.compose, this copies it to the custom docker container

#ADD ./nginx/certs /etc/nginx/certs/self-signed
#RUN apk add --no-cache git

RUN mkdir -p /var/www/html

# RUN rm -r /var/www/html/engine/tmp && mkdir /var/www/html/engine/tmp && chmod -R 777 /var/www/html/engine/tmp && chmod -R 777 /var/www/html/engine/files
