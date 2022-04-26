import fileinput
import random
import time

import numpy as np

# Python program for implementation of MergeSort with list
def mergeSort(arr):
    if len(arr) > 1:

        # Finding the mid of the array
        mid = len(arr)//2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        # Copy data to temp arrays L[] and R[]
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

#implementation of mergesort with numpy
#somethng is wrong. Use function merge_sort_np instead
def mergeSortNum(arr: np.ndarray):

    if arr.size > 1:

        # Finding the mid of the array
        mid = arr.size//2

        # Dividing the array elements
        L = arr[:mid]

        # into 2 halves
        R = arr[mid:]

        # Sorting the first half
        mergeSort(L)

        # Sorting the second half
        mergeSort(R)

        i = j = k = 0

        result=np.empty(arr.size,dtype=np.ndarray)    #will store the result array

        # Copy data to temp arrays L[] and R[]
        while i < L.size and j < R.size:
            if L[i] < R[j]:
                result[k] = L[i]
                i += 1
            else:
                result[k] = R[j]
                j += 1
            k += 1

        # Checking if any element was left
        while i < L.size:
            result[k] = L[i]
            i += 1
            k += 1

        while j < R.size:
            result[k] = R[j]
            j += 1
            k += 1

        return result

def mergeSort_np(arr: np.int):
    """
    Implementation of merge sort algorithm with numpy. Sorts an array of integers
    """
    data_length = np.shape(arr)[0]

    if data_length < 2:
        return arr

    midpoint = data_length // 2
    left = mergeSort_np(arr[:midpoint])
    right = mergeSort_np(arr[midpoint:])

    result = np.empty(data_length,dtype=np.int)
    l = r = 0
    k = 0
    r_length = len(right)
    l_length = len(left)

    while l < l_length and r < r_length:
        if left[l] <= right[r]:
            result[k] = left[l]
            l += 1
        else:
            result[k] = right[r]
            r += 1
        k += 1

    while l < l_length:
        result[k] = left[l]
        l += 1
        k += 1

    while r < r_length:
        result[k] = right[r]
        r += 1
        k += 1

    return result

# Code to print the list
def printList(arr):
    for i in range(len(arr)):
        print(arr[i], end=" ")
    print()

#Code to print the array
def printArr(arr):
    for i in range(arr.size):
        print(arr[i], end=" ")
    print()



def main():

    for line in fileinput.input():
        if len(line.rstrip()) == 0:
            print('Usage: {} <size of array>')
            exit(1)
        else:
            try:
                arrSize= int(line.rstrip())

            except ValueError as e:
                print('Integer convertion error: {}' .format(e))
                exit(2)

        if arrSize <= 0:
            print('Steps cannot be non-positive.')
            exit(3)

        arr=np.zeros(arrSize, dtype=np.int)    #initialize array of size arrSize

        for i in range (arrSize):
             arr[i]=random.randint(10, 100)

        # print("Given array is", end="\n")
        # printArr(arr)

        start_time = time.time()
        result=mergeSort_np(arr)
        stop_time = time.time() - start_time
        print ("Time %s sec " % stop_time)

        # print("Sorted array is: ", end="\n")
        # printArr(result)

        f=open("mergesort-Output.txt","a")
        text=['Sequential program',' input size ',str(arrSize),' time:',str(stop_time),"\n"]
        s=' '.join(text)
        f.write(s)
        f.close()






# Driver Code
if __name__ == '__main__':
    # arr = [12, 11, 13, 5, 6, 7]
    # print("Given array is", end="\n")
    # printList(arr)
    # mergeSort(arr)
    # print("Sorted array is: ", end="\n")
    # printList(arr)

    # a1D = np.array([12, 11, 13, 5, 6, 7],dtype=np.int)
    # print("Given array is", end="\n")
    # printArr(a1D)
    # # result=mergeSortNum(a1D)
    # result=mergeSort_np(a1D)
    # print("Sorted array is: ", end="\n")
    # printArr(result)
    main()




