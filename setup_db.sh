#!/bin/bash

psql "user=icpostgresql host=$POSTGRES_HOST port=5432 dbname=postgres target_session_attrs=read-write" <<SQL
\set pass `echo $POSTGRES_PASS`
CREATE DATABASE langgraph;
CREATE USER langgraph WITH PASSWORD :'pass';
GRANT ALL PRIVILEGES ON DATABASE "langgraph" to langgraph;
\connect langgraph
GRANT ALL ON SCHEMA public TO langgraph;
SQL
