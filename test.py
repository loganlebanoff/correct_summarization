# import sys
# from scoop import futures
# from absl import flags
# from absl import app
#
# if 'a' not in flags.FLAGS:
#     flags.DEFINE_string('a', 'x', 'description')
#
# FLAGS = flags.FLAGS
# FLAGS(sys.argv)
#
# apple = FLAGS.a
#
# def do(a):
#     try:
#         return FLAGS.a
#     except:
#         FLAGS(sys.argv)
#         return FLAGS.a
#
# def main(unused_argv):
#     if len(unused_argv) != 1: # prints a message if you've entered flags incorrectly
#         raise Exception("Problem with flags: %s" % unused_argv)
#     args = sys.argv[1:]
#     global apple
#     if 'both' in args:
#         apple = 'both'
#     ex_list = [1] * 1000
#     all_instances = list(futures.map(do, ex_list))
#     with open('data/asdf.txt', 'w') as f:
#         for x in all_instances:
#             f.write(x + '\n')
#     # print do()
#
# if __name__ == '__main__':
#
#     app.run(main)
#     # main()




import sys


from dask.distributed import Client
client = Client('10.173.204.48:8786')


def square(x):
    print x
    return x ** 2

def neg(x):
    return -x

[future] = client.scatter([range(1000000)], broadcast=True)
A = client.map(square, future)
print 'finished a'
B = client.map(neg, A)
total = client.submit(sum, B)
c = total.result()
print c