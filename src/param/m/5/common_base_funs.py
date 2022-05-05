import os
import time


def execmd(cmd):
    pipe = os.popen(cmd)
    re_content = pipe.read()
    pipe.close()
    return re_content


def mv(file1, file2):
    execmd(' '.join(['mv', file1, file2]))


def diff(file1, file2):
    return execmd(' '.join(['diff', file1, file2]))


def execmd_limit_time(cmd, timeout):
    cmd = ' '.join(['timeout', str(timeout), cmd])
    start = time.time()
    execmd(cmd)
    end = time.time()
    return (end - start) < int(timeout)


def get_file_content(path):
    f = open(path, 'r')
    content = f.read()
    f.close()
    return content


def get_file_lines(path):
    c = get_file_content(path)
    if c == '':
        return ''
    if c[-1] == '\n':
        return c[:-1].split('\n')
    else:
        return c.split('\n')


def put_file_content(path, content):
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()


def rm_file(f):
    if f is None:
        return
    if '/' == f or '/*' == f or '*' in f:
        return
    cmd = 'rm -rf ' + f
    execmd(cmd)


def mkdir_if_not_exists(d):
    if not os.path.exists(d):
        cmd = 'mkdir ' + d
        execmd(cmd)


def mkdir_p_if_not_exists(d):
    assert len(d) <= 50, 'd is too long! '
    if not os.path.exists(d):
        cmd = 'mkdir -p ' + d
        execmd(cmd)


def add_sep(d):
    if d[-1] != '/' and d[-1] != '\\':
        return d + os.sep
    return d


def get_last_dir(dir_name):
    t = 0
    last_d = None
    if os.path.exists(dir_name):
        dirs = os.listdir(dir_name)
        dirs = [add_sep(dir_name) + d for d in dirs]
        dirs = [d for d in dirs if os.path.isdir(d)]
        for d in dirs:
            mt = os.path.getmtime(d)
            if mt > t:
                t = mt
                last_d = d
    return last_d


def log(log_file, content):
    from main_configure_approach import MainProcessConf
    put_file_content(log_file, content)
    put_file_content(MainProcessConf.log_total, content)
