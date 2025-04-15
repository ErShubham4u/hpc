#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
using namespace std;

class Traverse {
private:
    vector<vector<int>> adjMatrix;
    vector<int> visited;
    int n;

public:
    void input() {
        cout << "Enter the number of vertices: ";
        cin >> n;
        adjMatrix.resize(n, vector<int>(n, 0));
        visited.resize(n, 0);

        cout << "Enter the adjacency matrix:\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cin >> adjMatrix[i][j];
            }
        }
    }

    void display() {
        cout << "Adjacency Matrix:\n";
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                cout << adjMatrix[i][j] << " ";
            }
            cout << endl;
        }
    }

    void bfs(int start) {
        queue<int> q;
        fill(visited.begin(), visited.end(), 0);

        #pragma omp parallel
        {
            #pragma omp single
            {
                q.push(start);
                visited[start] = 1;
                cout << "Parallel BFS Order: ";

                while (!q.empty()) {
                    int current = q.front();
                    q.pop();
                    cout << current << " ";

                    #pragma omp task firstprivate(current)
                    {
                        for (int i = 0; i < n; i++) {
                            if (adjMatrix[current][i] && !visited[i]) {
                                #pragma omp critical
                                {
                                    if (!visited[i]) {
                                        q.push(i);
                                        visited[i] = 1;
                                    }
                                }
                            }
                        }
                    }
                }

                cout << endl;
            }
        }
    }

    void dfs(int start) {
        stack<int> st;
        fill(visited.begin(), visited.end(), 0);

        #pragma omp parallel
        {
            #pragma omp single
            {
                st.push(start);
                visited[start] = 1;
                cout << "Parallel DFS Order: ";

                while (!st.empty()) {
                    int current = st.top();
                    st.pop();
                    cout << current << " ";

                    #pragma omp task firstprivate(current)
                    {
                        for (int i = n - 1; i >= 0; i--) {
                            if (adjMatrix[current][i] && !visited[i]) {
                                #pragma omp critical
                                {
                                    if (!visited[i]) {
                                        st.push(i);
                                        visited[i] = 1;
                                    }
                                }
                            }
                        }
                    }
                }

                cout << endl;
            }
        }
    }
};

int main() {
    Traverse t;
    t.input();
    t.display();

    int startVertex;
    cout << "Enter the starting vertex for traversal: ";
    cin >> startVertex;

    cout << "\nPerforming Parallel BFS:\n";
    t.bfs(startVertex);

    cout << "\nPerforming Parallel DFS:\n";
    t.dfs(startVertex);

    return 0;
}
