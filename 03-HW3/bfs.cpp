#include "bfs.h"

#include <cstdlib>
#include <omp.h>
#include <vector>

#include "../common/graph.h"

#ifdef VERBOSE
#include "../common/CycleTimer.h"
#include <stdio.h>
#endif // VERBOSE

constexpr int ROOT_NODE_ID = 0;
constexpr int NOT_VISITED_MARKER = -1;

void vertex_set_clear(VertexSet *list)
{
    list->count = 0;
}

void vertex_set_init(VertexSet *list, int count)
{
    list->max_vertices = count;
    list->vertices = new int[list->max_vertices];
    vertex_set_clear(list);
}

void vertex_set_destroy(VertexSet *list)
{
    delete[] list->vertices;
}

// Take one step of "top-down" BFS.  For each vertex on the frontier,
// follow all outgoing edges, and add all neighboring vertices to the
// new_frontier.
void top_down_step(Graph g, VertexSet *frontier, VertexSet *new_frontier, int *distances)
{
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < frontier->count; i++)
    {

        int node = frontier->vertices[i];

        int start_edge = g->outgoing_starts[node];
        int end_edge = (node == g->num_nodes - 1) ? g->num_edges : g->outgoing_starts[node + 1];

        // attempt to add all neighbors to the new frontier
        for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
        {
            int outgoing = g->outgoing_edges[neighbor];

            if (distances[outgoing] == NOT_VISITED_MARKER)
            {
                if (__sync_bool_compare_and_swap(&distances[outgoing], NOT_VISITED_MARKER, distances[node] + 1))
                {
                    int index = __sync_fetch_and_add(&new_frontier->count, 1);
                    new_frontier->vertices[index] = outgoing;
                }
            }
        }
    }
}

// Implements top-down BFS.
//
// Result of execution is that, for each node in the graph, the
// distance to the root is stored in sol.distances.
void bfs_top_down(Graph graph, solution *sol)
{

    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);

    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    // initialize all nodes to NOT_VISITED
#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
        sol->distances[i] = NOT_VISITED_MARKER;

    // setup frontier with the root node
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;
    sol->distances[ROOT_NODE_ID] = 0;

    while (frontier->count != 0)
    {

#ifdef VERBOSE
        double start_time = CycleTimer::current_seconds();
#endif

        vertex_set_clear(new_frontier);

        top_down_step(graph, frontier, new_frontier, sol->distances);

#ifdef VERBOSE
        double end_time = CycleTimer::current_seconds();
        printf("frontier=%-10d %.4f sec\n", frontier->count, end_time - start_time);
#endif

        // swap pointers
        VertexSet *tmp = frontier;
        frontier = new_frontier;
        new_frontier = tmp;
    }

    // free memory
    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
}

void bottom_up_step(Graph g, bool *frontier, bool *new_frontier, int *distances, int iteration)
{
#pragma omp parallel for schedule(dynamic, 1024)
    for (int i = 0; i < g->num_nodes; i++)
    {
        if (distances[i] == NOT_VISITED_MARKER)
        {
            int start_edge = g->incoming_starts[i];
            int end_edge = (i == g->num_nodes - 1) ? g->num_edges : g->incoming_starts[i + 1];
            for (int neighbor = start_edge; neighbor < end_edge; neighbor++)
            {
                int incoming = g->incoming_edges[neighbor];
                if (frontier[incoming])
                {
                    distances[i] = iteration;
                    new_frontier[i] = true;
                    break;
                }
            }
        }
    }
}

void bfs_bottom_up(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "bottom up" BFS here as
    // described in the handout.
    //
    // As a result of your code's execution, sol.distances should be
    // correctly populated for all nodes in the graph.
    //
    // As was done in the top-down case, you may wish to organize your
    // code by creating subroutine bottom_up_step() that is called in
    // each step of the BFS process.
    bool *frontier = new bool[graph->num_nodes];
    bool *new_frontier = new bool[graph->num_nodes];
    int frontier_count = 1;

#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier[i] = false;
        new_frontier[i] = false;
    }

    sol->distances[ROOT_NODE_ID] = 0;
    frontier[ROOT_NODE_ID] = true;

    int iteration = 1;
    while (frontier_count > 0)
    {
        frontier_count = 0;
        bottom_up_step(graph, frontier, new_frontier, sol->distances, iteration);
#pragma omp parallel for reduction(+ \
                                   : frontier_count)
        for (int i = 0; i < graph->num_nodes; i++)
        {
            frontier[i] = new_frontier[i];
            new_frontier[i] = false;
            if (frontier[i])
            {
                frontier_count++;
            }
        }
        iteration++;
    }
    delete[] frontier;
    delete[] new_frontier;
}

void bfs_hybrid(Graph graph, solution *sol)
{
    // For PP students:
    //
    // You will need to implement the "hybrid" BFS here as
    // described in the handout.
    VertexSet list1;
    VertexSet list2;
    vertex_set_init(&list1, graph->num_nodes);
    vertex_set_init(&list2, graph->num_nodes);
    VertexSet *frontier = &list1;
    VertexSet *new_frontier = &list2;

    bool *frontier_map = new bool[graph->num_nodes];
    bool *new_frontier_map = new bool[graph->num_nodes];

#pragma omp parallel for
    for (int i = 0; i < graph->num_nodes; i++)
    {
        sol->distances[i] = NOT_VISITED_MARKER;
        frontier_map[i] = false;
    }

    sol->distances[ROOT_NODE_ID] = 0;
    frontier->vertices[frontier->count++] = ROOT_NODE_ID;

    bool use_top_down = true;
    int iteration = 1;

    while (frontier->count > 0)
    {
        if (use_top_down)
        {
            long long mf = 0;
#pragma omp parallel for reduction(+ : mf)
            for (int i = 0; i < frontier->count; i++)
            {
                mf += outgoing_size(graph, frontier->vertices[i]);
            }
            if (mf > graph->num_edges / 24)
            {
                use_top_down = false;
#pragma omp parallel for
                for (int i = 0; i < frontier->count; i++)
                {
                    frontier_map[frontier->vertices[i]] = true;
                }
            }
        }
        else if (frontier->count < graph->num_nodes / 24)
        {
            use_top_down = true;
#pragma omp parallel for
            for(int i = 0; i < graph->num_nodes; i++) {
                if(sol->distances[i] == iteration - 1) {
                    int index = __sync_fetch_and_add(&frontier->count, 1);
                    frontier->vertices[index] = i;
                }
            }
        }

        if (use_top_down)
        {
            vertex_set_clear(new_frontier);
            top_down_step(graph, frontier, new_frontier, sol->distances);
            VertexSet *tmp = frontier;
            frontier = new_frontier;
            new_frontier = tmp;
        }
        else
        {
#pragma omp parallel for
            for (int i = 0; i < graph->num_nodes; i++)
            {
                new_frontier_map[i] = false;
            }
            bottom_up_step(graph, frontier_map, new_frontier_map, sol->distances, iteration);

            int new_frontier_count = 0;
#pragma omp parallel for reduction(+ : new_frontier_count)
            for (int i = 0; i < graph->num_nodes; i++)
            {
                frontier_map[i] = new_frontier_map[i];
                if (frontier_map[i])
                {
                    new_frontier_count++;
                }
            }
            frontier->count = new_frontier_count;
        }
        iteration++;
    }

    vertex_set_destroy(&list1);
    vertex_set_destroy(&list2);
    delete[] frontier_map;
    delete[] new_frontier_map;
}