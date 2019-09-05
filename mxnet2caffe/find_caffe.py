try:
    import caffe
except ImportError:
    import sys
    caffe_python_root = '/home/lfx/Tool/caffe_cpu/python'
    sys.path.append(caffe_python_root)
    try:
        import caffe
    except ImportError:
        print('No module named caffe, ' + \
              'please reset "caffe_python_root" in ' + __file__)
        exit(-1)
    sys.path.pop()
