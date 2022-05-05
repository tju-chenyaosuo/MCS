import random
import common_base_funs

if __name__ == '__main__':
    seed_list = [str(random.randint(0, 80000000)) for _ in range(1000000)]
    common_base_funs.put_file_content('seed.txt', '\n'.join(seed_list))
