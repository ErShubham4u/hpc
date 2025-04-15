#include <iostream> 
#include <vector> 
#include <omp.h> 
#include <ctime> 
using namespace std; 
 
void printArray(const vector<int>& arr) { 
    for (int num : arr) { 
        cout << num << " "; 
    } 
    cout << endl; 
} 
 
void bubbleSortSequential(vector<int>& arr) { 
    int n = arr.size(); 
    for (int i = 0; i < n - 1; i++) { 
        for (int j = 0; j < n - i - 1; j++) { 
            if (arr[j] > arr[j + 1]) { 
                swap(arr[j], arr[j + 1]); 
            } 
        } 
    } 
} 
 
void bubbleSortParallel(vector<int>& arr) { 
    int n = arr.size(); 
    #pragma omp parallel for 
    for (int i = 0; i < n - 1; i++) { 
        for (int j = 0; j < n - i - 1; j++) { 
            if (arr[j] > arr[j + 1]) { 
                swap(arr[j], arr[j + 1]); 
            } 
        } 
    } 
} 
 
void merge(vector<int>& arr, int left, int mid, int right) { 
    int n1 = mid - left + 1; 
    int n2 = right - mid; 
    vector<int> L(n1), R(n2); 
 
    for (int i = 0; i < n1; i++) 
        L[i] = arr[left + i]; 
    for (int j = 0; j < n2; j++) 
        R[j] = arr[mid + 1 + j]; 
 
    int i = 0, j = 0, k = left; 
    while (i < n1 && j < n2) { 
        if (L[i] <= R[j]) { 
            arr[k] = L[i]; 
            i++; 
        } else { 
            arr[k] = R[j]; 
            j++; 
        } 
        k++; 
    } 
 
    while (i < n1) { 
        arr[k] = L[i]; 
        i++; 
        k++; 
    } 
 
    while (j < n2) { 
        arr[k] = R[j]; 
        j++; 
        k++; 
    } 
} 
 
void mergeSortSequential(vector<int>& arr, int left, int right) { 
    if (left < right) { 
        int mid = left + (right - left) / 2; 
        mergeSortSequential(arr, left, mid); 
        mergeSortSequential(arr, mid + 1, right); 
        merge(arr, left, mid, right); 
    } 
} 
 
void mergeSortParallel(vector<int>& arr, int left, int right) { 
    if (left < right) { 
        int mid = left + (right - left) / 2; 
 
        #pragma omp parallel sections 
        { 
            #pragma omp section 
            mergeSortParallel(arr, left, mid); 
 
            #pragma omp section 
            mergeSortParallel(arr, mid + 1, right); 
        } 
 
        merge(arr, left, mid, right); 
    } 
} 
 
int main() { 
    int n; 
    cout << "Enter number of elements: "; 
    cin >> n; 
     
    vector<int> arr(n); 
    cout << "Enter elements: "; 
    for (int i = 0; i < n; i++) { 
        cin >> arr[i]; 
    } 
 
    vector<int> arrCopy = arr; 
    double start = omp_get_wtime(); 
    bubbleSortSequential(arrCopy); 
    double end = omp_get_wtime(); 
    cout << "Sequential Bubble Sort Time: " << (end - start) << " seconds" << endl; 
    cout << "Sorted Array (Bubble Sort - Sequential): "; 
    printArray(arrCopy); 
 
    arrCopy = arr; 
    start = omp_get_wtime(); 
    bubbleSortParallel(arrCopy); 
    end = omp_get_wtime(); 
    cout << "Parallel Bubble Sort Time: " << (end - start) << " seconds" << endl; 
    cout << "Sorted Array (Bubble Sort - Parallel): "; 
    printArray(arrCopy); 
 
    arrCopy = arr; 
    start = omp_get_wtime(); 
    mergeSortSequential(arrCopy, 0, n - 1); 
    end = omp_get_wtime(); 
    cout << "Sequential Merge Sort Time: " << (end - start) << " seconds" << endl; 
    cout << "Sorted Array (Merge Sort - Sequential): "; 
    printArray(arrCopy); 
 
    arrCopy = arr; 
    start = omp_get_wtime(); 
    mergeSortParallel(arrCopy, 0, n - 1); 
    end = omp_get_wtime(); 
    cout << "Parallel Merge Sort Time: " << (end - start) << " seconds" << endl; 
    cout << "Sorted Array (Merge Sort - Parallel): "; 
    printArray(arrCopy); 
 
    return 0; 
}