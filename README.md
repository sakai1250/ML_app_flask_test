# ML_app_flask_test
https://github.com/sakai1250/ML_app_flask_test/assets/92532910/1be15b1e-99e3-47f7-90ce-6099047c9707


```bash:(terminal)
$git clone https://github.com/sakai1250/ML_app_flask_test
$cd ML_app_flask_test && python detect_goldfish.py
$docker build -t flask-app .
$docker run -p 5000:5000 flask-app
```
if you fail with 
```bash:(terminal)
$docker run -p 5000:5000 flask-app
```
then, use

```bash:(terminal)
$cd app && python app.py
```
