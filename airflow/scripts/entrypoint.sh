#!/bin/bash
airflow db init
airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email maxwellselassie2004@gmail.com \
  --password itzmcs2004 || true

exec airflow webserver &
exec airflow scheduler
