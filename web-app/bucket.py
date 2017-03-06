bucket_list = {0.25: 1, 0.5: 2, 0.75: 3, 1: 4, 1.25: 5, 1.5: 6, 1.75: 7,
               2: 8, 2.25: 9, 2.5: 10, 2.75: 11, 3: 12, 3.25: 13, 3.5: 14,
               3.75: 15, 4: 16, 4.25: 17, 4.5: 18, 4.75: 19, 5: 20, 5.25: 21, 5.5: 22, 5.75: 23, 6: 24, 6.25: 25, 6.5: 26, 6.75: 27, 7: 28, 7.25: 29, 7.5: 30, 7.75: 31, 8: 32, 8.25: 33, 8.5: 34, 8.75: 35, 9: 36, 9.25: 37, 9.5: 38, 9.75: 39, 10: 40}

bucket_list_new = sorted(bucket_list.iteritems(), key=lambda (x, y): float(x))
new_bucket = []


def create_bucket(a):
    new_bucket = []
    for i in a:
        for j in bucket_list_new:
            if(j[0] >= i):
                new_bucket.append(j[1])
                break
    return new_bucket

def bucket_to_bandgap_conversion(b):
#    print(b)
    for key,value in bucket_list.iteritems():
        if(value==b):
             return key
        
