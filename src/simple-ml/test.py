import os

tar_dir = 'test_dir'

cra_cmd = ' '.join(['find', tar_dir, '-name', "'*cra*'"])
mis_cmd = ' '.join(['find', tar_dir, '-name', "'*mis*'"])
time_cmd = ' '.join(['find', tar_dir, '-name', "'*time*'"])


def rm_file(f):
    if f is None:
        return
    if '/' == f or '/*' == f or '*' in f:
        return
    cmd = 'rm -rf ' + f
    print(cmd)
    execmd(cmd)


def execmd(cmd):
    pipe = os.popen(cmd)
    re_content = pipe.read()
    pipe.close()
    return re_content


if __name__ == '__main__':
    cras = execmd(cra_cmd).split('\n')
    cras = [_ for _ in cras if len(_) > 1]
    miss = execmd(mis_cmd).split('\n')
    miss = [_ for _ in miss if len(_) > 1]
    times = execmd(time_cmd).split('\n')
    times = [_ for _ in times if len(_) > 1]
    remains = cras + miss + times
    remains = [_.split('/')[1] for _ in remains]
    alls = os.listdir(tar_dir)
    removes = list(set(alls) - set(remains))

    # print(len(cras))
    # print(len(miss))
    # print(len(times))
    # print(len(os.listdir(tar_dir)))
    # print(remains[0])
    # print(alls[0])

    for _ in removes:
        tar = tar_dir + os.sep + _

        cra_cmd = ' '.join(['find', tar, '-name', "'*cra*'"])
        mis_cmd = ' '.join(['find', tar, '-name', "'*mis*'"])
        time_cmd = ' '.join(['find', tar, '-name', "'*time*'"])
        cras = execmd(cra_cmd).split('\n')
        cras = [_ for _ in cras if len(_) > 1]
        miss = execmd(mis_cmd).split('\n')
        miss = [_ for _ in miss if len(_) > 1]
        times = execmd(time_cmd).split('\n')
        times = [_ for _ in times if len(_) > 1]
        remains = cras + miss + times

        assert len(remains) == 0

        rm_file(tar)
    pass
