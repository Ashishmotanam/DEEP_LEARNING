import tensorflow as tf
#creaated a matrix to store values of mat1(a)
mat1=tf.constant([[2,4],[3,5]])

#creaated a matrix to store values of mat2(b)
mat2=tf.constant([[1,2],[2,4]])

#creaated a matrix to store values of mat3(c)
mat3=tf.constant([[4,5],[3,2]])
#performing the function mat1 power 2(a^2)
power_mat=tf.matmul(mat1,mat1)
#adding result power_mat to tp mat2(a^2 +b )
mat_add=tf.add(power_mat,mat2)
#multiplying the result mat_add to mat3(a^2 + b)*c
mat_mul=tf.matmul(mat_add,mat3)
sess = tf.Session()
#running a session on mat_mul
with tf.Session() as sess:
    print(sess.run(mat_mul))
sess.close()