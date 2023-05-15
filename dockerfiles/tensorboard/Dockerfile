FROM python:3.8

RUN pip3 --no-cache-dir install --upgrade \
    tensorboard==2.10.1 \
    torch_tb_profiler==0.4.0

CMD ["tensorboard", "--bind_all", "--logdir", "/outputs"]
