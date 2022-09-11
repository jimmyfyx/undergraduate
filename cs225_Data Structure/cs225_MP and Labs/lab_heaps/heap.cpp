
/**
 * @file heap.cpp
 * Implementation of a heap class.
 */

template <class T, class Compare>
size_t heap<T, Compare>::root() const
{
    // @TODO Update to return the index you are choosing to be your root.
    return 1;
}

template <class T, class Compare>
size_t heap<T, Compare>::leftChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the left child.
    return currentIdx * 2;
}

template <class T, class Compare>
size_t heap<T, Compare>::rightChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the right child.
    return currentIdx * 2 + 1;
}

template <class T, class Compare>
size_t heap<T, Compare>::parent(size_t currentIdx) const
{
    // @TODO Update to return the index of the parent.
    return currentIdx / 2;
}

template <class T, class Compare>
bool heap<T, Compare>::hasAChild(size_t currentIdx) const
{
    // @TODO Update to return whether the given node has a child
    if (currentIdx * 2 >= _elems.size() && currentIdx * 2 + 1 >= _elems.size())
    {
        return false;
    }
    return true;
}

template <class T, class Compare>
size_t heap<T, Compare>::maxPriorityChild(size_t currentIdx) const
{
    // @TODO Update to return the index of the child with highest priority
    ///   as defined by higherPriority()
    if (rightChild(currentIdx) >= _elems.size())
    {
        return leftChild(currentIdx);
    }

    if (higherPriority(_elems[leftChild(currentIdx)], _elems[rightChild(currentIdx)]) == true)
    {
        return leftChild(currentIdx);
    }
    else
    {
        return rightChild(currentIdx);
    }
}

template <class T, class Compare>
void heap<T, Compare>::heapifyDown(size_t currentIdx)
{
    // @TODO Implement the heapifyDown algorithm.
    if (hasAChild(currentIdx) == false)
    {
        return;
    }

    size_t max_child = maxPriorityChild(currentIdx);

    if (higherPriority(_elems[max_child], _elems[currentIdx]) == true)
    {
        std::swap(_elems[currentIdx], _elems[max_child]);
        heapifyDown(max_child);
    }
}

template <class T, class Compare>
void heap<T, Compare>::heapifyUp(size_t currentIdx)
{
    if (currentIdx == root())
        return;
    size_t parentIdx = parent(currentIdx);
    if (higherPriority(_elems[currentIdx], _elems[parentIdx])) {
        std::swap(_elems[currentIdx], _elems[parentIdx]);
        heapifyUp(parentIdx);
    }
}

template <class T, class Compare>
heap<T, Compare>::heap()
{
    // @TODO Depending on your implementation, this function may or may
    ///   not need modifying
    T inf = numeric_limits<T>::min();
    _elems.push_back(inf);
}


template <class T, class Compare>
heap<T, Compare>::heap(const std::vector<T>& elems) 
{
    // @TODO Construct a heap using the buildHeap algorithm
    T inf = numeric_limits<T>::min();
    _elems.push_back(inf);

    for (unsigned int i = 0; i < elems.size(); i ++)
    {
        _elems.push_back(elems[i]);
    }
    // _elems.insert(_elems.end(), elems.begin(), elems.end());
    for (size_t i = _elems.size() - 1; i > 0; i --)
    {
        heapifyDown(i);
    }
}

template <class T, class Compare>
T heap<T, Compare>::pop()
{
    // @TODO Remove, and return, the element with highest priority
    if (empty() == false) 
    {
        size_t size = _elems.size();

        T p = _elems[1];
        _elems[1] = _elems[size - 1];
        _elems.pop_back();
        heapifyDown(1);
        return p;
    }

    return T();
}

template <class T, class Compare>
T heap<T, Compare>::peek() const
{
    // @TODO Return, but do not remove, the element with highest priority
    return _elems[root()];
}

template <class T, class Compare>
void heap<T, Compare>::push(const T& elem)
{
    // @TODO Add elem to the heap
    _elems.push_back(elem);
    heapifyUp(_elems.size() - 1);
}

template <class T, class Compare>
void heap<T, Compare>::updateElem(const size_t & idx, const T& elem)
{
    // @TODO In-place updates the value stored in the heap array at idx
    // Corrects the heap to remain as a valid heap even after update
    _elems[idx] = elem;

    if (higherPriority(_elems[idx], _elems[parent(idx)]) == true)
    {
        heapifyUp(idx);
    }
    else
    {
        // possible to heapify down
        if (hasAChild(idx) == true)
        {
            size_t max_child = maxPriorityChild(idx);
            if (higherPriority(_elems[max_child], _elems[idx]) == true)
            {
                std::swap(_elems[idx], _elems[max_child]);
                heapifyDown(max_child);
            }
        }
    }
}


template <class T, class Compare>
bool heap<T, Compare>::empty() const
{
    // @TODO Determine if the heap is empty
    if (_elems.size() <= 1)
    {
        return true;
    }
    return false;
}

template <class T, class Compare>
void heap<T, Compare>::getElems(std::vector<T> & heaped) const
{
    for (size_t i = root(); i < _elems.size(); i++) {
        heaped.push_back(_elems[i]);
    }
}
