#!/usr/bin/env python
# coding: utf-8

import requests

url = 'http://localhost:8888/predict'


customer = {
    "age": 18,
    "sex": "male",
    "bmi": 33.77,
    "children": 1,
    "smoker": "non-smoke",
    "region": 2,
    "charges": 1725.5523,
    "insuranceclaim": "yes"
}

insuranceclaim = requests.post(url, json=customer).json()
print(insuranceclaim)

if insuranceclaim['insuranceclaim']:
    print('Customer is interested to Claimed Health Insurance')
else:
    print('Customer is Not interested to Claimed Health Insurance')
