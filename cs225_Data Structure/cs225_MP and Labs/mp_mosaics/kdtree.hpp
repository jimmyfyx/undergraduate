/**
 * @file kdtree.cpp
 * Implementation of KDTree class.
 */

#include <utility>
#include <algorithm>

using namespace std;

template <int Dim>
bool KDTree<Dim>::smallerDimVal(const Point<Dim>& first,
                                const Point<Dim>& second, int curDim) const
{
    /**
     * @todo Implement this function!
     */
  if (first[curDim] != second[curDim])
  {
    if (first[curDim] < second[curDim])
    {
      return true;
    }
    else
    {
      return false;
    }
  }
  else
  {
    // Use operator < to break the tie
    return first < second;
  }
  return false;
}

template <int Dim>
bool KDTree<Dim>::shouldReplace(const Point<Dim>& target,
                                const Point<Dim>& currentBest,
                                const Point<Dim>& potential) const
{
    /**
     * @todo Implement this function!
     */
  double distance_cur = 0;
  double distance_pot = 0;
  for (int i = 0; i < Dim; i++)
  {
    distance_cur = distance_cur + (target[i] - currentBest[i]) * (target[i] - currentBest[i]);
    distance_pot = distance_pot + (target[i] - potential[i]) * (target[i] - potential[i]);
  }

  if (distance_pot < distance_cur)
  {
    return true;
  }
  else if (distance_pot > distance_cur)
  {
    return false;
  }
  else
  {
    return potential < currentBest;
  }
  return false;
}

template <int Dim>
KDTree<Dim>::KDTree(const vector<Point<Dim>>& newPoints)
{
    /**
     * @todo Implement this function!
     */
  vector<Point<Dim>> newPoints_copy;
  for (unsigned int i = 0; i < newPoints.size(); i++)
  {
    newPoints_copy.push_back(newPoints[i]);
  }

  int left = 0;
  int right = newPoints.size() - 1;

  root = BuildTree(newPoints_copy, left, right, 0);
  size = newPoints.size();
}

template <int Dim>
KDTree<Dim>::KDTree(const KDTree<Dim>& other) {
  /**
   * @todo Implement this function!
   */
  root = BuildTree_copy(other.root);
  size = other.size;
}

template <int Dim>
const KDTree<Dim>& KDTree<Dim>::operator=(const KDTree<Dim>& rhs) {
  /**
   * @todo Implement this function!
   */
  DestroyTree(root);
  root = BuildTree_copy(rhs.root);
  size = rhs.size;

  return *this;
}

template <int Dim>
KDTree<Dim>::~KDTree() {
  /**
   * @todo Implement this function!
   */
  DestroyTree(root);
}

template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor(const Point<Dim>& query) const
{
    /**
     * @todo Implement this function!
     */
  return findNearestNeighbor_help(query, root, 0);
}


// Helper function findNearestNeighbor_help
template <int Dim>
Point<Dim> KDTree<Dim>::findNearestNeighbor_help(const Point<Dim>& query, KDTreeNode *root, int Dimension) const
{
  if (root -> left == NULL && root -> right == NULL)
  {
    return root -> point;
  }

  Point <Dim> curBest = root -> point;

  bool wentleft = false;
  if (smallerDimVal(query, root -> point, Dimension) == true)
  {
    if (root -> left != NULL)
    {
      curBest = findNearestNeighbor_help(query, root -> left, (Dimension + 1) % Dim);
      wentleft = true;
    }
  }
  else if (smallerDimVal(query, root -> point, Dimension) == false)
  {
    if (root -> right != NULL)
    {
      curBest = findNearestNeighbor_help(query, root -> right, (Dimension + 1) % Dim);
      wentleft = false;
    }
  } 
  
  // During backtracking, check whether the current point is closer to the query
  if (shouldReplace(query, curBest, root -> point) == true)
  {
    curBest = root -> point;
  }

  // Construct the circle with radius distance between query and curBest
  double radius_square = 0;
  for (int i = 0; i < Dim; i++)
  {
    radius_square = radius_square + (query[i] - curBest[i]) * (query[i] - curBest[i]);
  }

  double split_dis = (root -> point[Dimension] - query[Dimension]) * (root -> point[Dimension] - query[Dimension]);


  Point<Dim> temp_best = curBest;
  if (radius_square >= split_dis)
  {
    // Step into the opposite subtree
    if (wentleft == true && root -> right != NULL)
    {
      temp_best = findNearestNeighbor_help(query, root -> right, (Dimension + 1) % Dim);
      if (shouldReplace(query, curBest, temp_best) == true)
      {
        curBest = temp_best;
        // Update the radius
        /*
        radius_square = 0;
        for (int i = 0; i < Dim; i++)
        {
          radius_square = radius_square + (query[i] - curBest[i]) * (query[i] - curBest[i]);
        }
        */
      }
    }
    
    if (wentleft == false && root -> left != NULL)
    {
      temp_best = findNearestNeighbor_help(query, root -> left, (Dimension + 1) % Dim);
      if (shouldReplace(query, curBest, temp_best) == true)
      {
        curBest = temp_best;
        // Update the radius
        /*
        radius_square = 0;
        for (int i = 0; i < Dim; i++)
        {
          radius_square = radius_square + (query[i] - curBest[i]) * (query[i] - curBest[i]);
        }
        */
      }
    }
  }

  return curBest;
}

// Helper function DestroyTree
template <int Dim>
void KDTree<Dim>::DestroyTree(KDTreeNode *curRoot)
{
  if (curRoot == NULL)
  {
    return;
  }

  DestroyTree(curRoot -> left);
  DestroyTree(curRoot -> right);
  delete curRoot;
}


// Helper function BuildTree_copy
template <int Dim>
typename KDTree<Dim>::KDTreeNode* KDTree<Dim>::BuildTree_copy(KDTreeNode *curRoot)
{
  if (curRoot == NULL)
  {
    return NULL;
  }

  // Create and initialize the new node
  KDTreeNode *new_node = new KDTreeNode();
  // Initialize the new point
  for (int i = 0; i < Dim; i++)
  {
    (new_node -> point)[i] = (curRoot -> point)[i];
  }

  // Recursive case
  new_node -> left = BuildTree_copy(curRoot -> left);
  new_node -> right = BuildTree_copy(curRoot -> right);

  return new_node;
}


// Helper function BuildTree
template <int Dim>
typename KDTree<Dim>::KDTreeNode* KDTree<Dim>::BuildTree(vector<Point<Dim>>& list, int left, int right, int Dimension)
{
  if (left <= right)
  {
    // Determine the median index
    int middle = (left + right) / 2;
    // Create a new tree node and initialize the node
    KDTreeNode *curRoot = new KDTreeNode(Quickselect(list, left, right, middle, Dimension));
    
    // Recursive case
    curRoot -> left = BuildTree(list, left, middle - 1, (Dimension + 1) % Dim);
    curRoot -> right = BuildTree(list, middle + 1, right, (Dimension + 1) % Dim);

    return curRoot;
  }
  return NULL;
}

// Helper function Quickselect
template <int Dim>
Point<Dim> KDTree<Dim>::Quickselect(vector<Point<Dim>>& list, int left, int right, int k, int Dimension)
{
  while (true)
  {
    if (left == right)
    {
      return list[left];
    }

    int pivot_index = k;
    pivot_index = Partition(list, left, right, pivot_index, Dimension);
    if (pivot_index == k)
    {
      return list[k];
    }
    else if (pivot_index > k)
    {
      right = pivot_index - 1;
    }
    else
    {
      left = pivot_index + 1;
    }
  }
}

// Helper function Partition
template <int Dim>
int KDTree<Dim>::Partition(vector<Point<Dim>>& list, int left, int right, int pivot_index, int Dimension)
{
  int pivotValue = list[pivot_index][Dimension];
  // Swap list[pivot_index] and list[right]
  Point<Dim> temp;
  
  temp = list[pivot_index];
  list[pivot_index] = list[right];
  list[right] = temp;

  int storeIndex = left;

  for (int i = left; i < right; i++)
  {
    if (list[i][Dimension] < pivotValue)
    {
      // Swap list[storeIndex] and list[i]
      temp = list[storeIndex];
      list[storeIndex] = list[i];
      list[i] = temp;
  
      storeIndex ++;
    }
    else if (list[i][Dimension] == pivotValue)
    {
      // Use operator < of points to break the tie
      if (list[i] < list[right])
      {
        // Swap list[storeIndex] and list[i]
        temp = list[storeIndex];
        list[storeIndex] = list[i];
        list[i] = temp;
      
        storeIndex ++;
      }
    }
  }

  // Swap list[right] and list[storeIndex]
  temp = list[storeIndex];
  list[storeIndex] = list[right];
  list[right] = temp;

  return storeIndex;
}

