/**
 * @file list.cpp
 * Doubly Linked List (MP 3).
 */
#include <iostream>
using namespace std;

template <class T>
List<T>::List() //: head_(NULL), tail_(NULL), length_(0) 
{ 
  head_ = NULL;
  tail_ = NULL;
  length_ = 0;
}

/**
 * Returns a ListIterator with a position at the beginning of
 * the List.
 */
template <typename T>
typename List<T>::ListIterator List<T>::begin() const {
  // @TODO: graded in MP3.1
  return List<T>::ListIterator(head_);
}

/**
 * Returns a ListIterator one past the end of the List.
 */
template <typename T>
typename List<T>::ListIterator List<T>::end() const {
  // @TODO: graded in MP3.1
  return List<T>::ListIterator(NULL);
}


/**
 * Destroys all dynamically allocated memory associated with the current
 * List class.
 */
template <typename T>
void List<T>::_destroy() {
  /// @todo Graded in MP3.1
  if (head_ == NULL && tail_ == NULL)
  {
    // Empty list
    return;
  }
  else
  {
    // First set up a pointer cur pointing to the first node
    // 'First Node' is the first node after head node
    ListNode *cur = head_ -> next;
    // 'temp' pointer initially also points to the first node
    ListNode *temp = head_ -> next;

    if (cur == NULL)
    {
      // The list only has one node
      delete head_;
    }
    else
    {
      // The list at least has two nodes
      while (cur != tail_)
      {
        // Get the next node
        temp = cur -> next;
        // Delete the current node
        delete cur;
        head_ -> next = temp;
        // Update the 'cur' pointer
        cur = temp;
      }

      // 'cur' points to the last node
      // Delete the last node
      delete cur;
      tail_ = NULL;
      // Only remains the head node
      // Delete the head node
      delete head_;
      head_ = NULL;
    }
  }
}

/**
 * Inserts a new node at the front of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <typename T>
void List<T>::insertFront(T const & ndata) {
  /// @todo Graded in MP3.1
  ListNode * newNode = new ListNode(ndata);
  newNode -> next = head_;
  newNode -> prev = NULL;
  
  if (head_ != NULL && tail_ != NULL) 
  {
    // The list is not empty
    head_ -> prev = newNode;
    // Update the head pointer
    head_ = newNode;
  }
  else if (head_ == NULL && tail_ == NULL) 
  {
    // The list is empty
    tail_ = newNode;
    // Update the head pointer
    head_ = newNode;
  }

  newNode = NULL;
  length_++;
}

/**
 * Inserts a new node at the back of the List.
 * This function **SHOULD** create a new ListNode.
 *
 * @param ndata The data to be inserted.
 */
template <typename T>
void List<T>::insertBack(const T & ndata) {
  /// @todo Graded in MP3.1
  ListNode * newNode = new ListNode(ndata);
  newNode -> next = NULL;
  newNode -> prev = tail_;

  if (head_ != NULL && tail_ != NULL) 
  {
    // The list is not empty
    tail_ -> next = newNode;
    // Update the tail pointer
    tail_ = newNode;
  }
  else if (head_ == NULL && tail_ == NULL) 
  {
    // The list is empty
    tail_ = newNode;
    // Update the head pointer
    head_ = newNode;
  }

  newNode = NULL;
  length_++;
}

/**
 * Helper function to split a sequence of linked memory at the node
 * splitPoint steps **after** start. In other words, it should disconnect
 * the sequence of linked memory after the given number of nodes, and
 * return a pointer to the starting node of the new sequence of linked
 * memory.
 *
 * This function **SHOULD NOT** create **ANY** new List or ListNode objects!
 *
 * This function is also called by the public split() function located in
 * List-given.hpp
 *
 * @param start The node to start from.
 * @param splitPoint The number of steps to walk before splitting.
 * @return The starting node of the sequence that was split off.
 */
template <typename T>
typename List<T>::ListNode * List<T>::split(ListNode * start, int splitPoint) {
  /// @todo Graded in MP3.1
  ListNode * curr = start;

  // First check whether 'start' points to valid node
  if (curr == NULL || curr -> next == NULL)
  {
    // Invalid starting node
    return start;
  }

  if (splitPoint == 0 || splitPoint >= length_)
  {
    // The splitPoint is out of range of the list
    // Also the condition for that there is no need to split
    return start;
  }

  // Use 'curr' to point to the head of the new list sequence
  for (int i = 0; i < splitPoint; i++) 
  {
    if (curr -> next != NULL)
    {
      curr = curr->next;
    }
    else
    {
      // The split point is invalid
      return curr;
    }
  }

  // Now 'curr' points to the head of the new list sequence
  if (curr != NULL) 
  {
    // Set the last node of the first sequence
    // tail_ = curr -> prev;
    curr -> prev -> next = NULL;
    // Set the node pointed by curr
    curr -> prev = NULL;
    return curr;
  }

  return curr;
}

/**
  * Modifies List using the rules for a TripleRotate.
  *
  * This function will to a wrapped rotation to the left on every three 
  * elements in the list starting for the first three elements. If the 
  * end of the list has a set of 1 or 2 elements, no rotation all be done 
  * on the last 1 or 2 elements.
  * 
  * You may NOT allocate ANY new ListNodes!
  */
template <typename T>
void List<T>::tripleRotate() {
  // @todo Graded in MP3.1
  if (length_ < 3)
  {
    // Less than three nodes in the list
    // No change
    return;
  }
  else
  {
    // >= 3 nodes in the list
    ListNode *curr_ = head_; 
    ListNode *prev_ = head_;
    ListNode *next_ = head_ -> next -> next;
    ListNode *before_ = NULL;
    ListNode *after_ = curr_ -> next -> next -> next;

    while(curr_ != NULL)
    {
      // Check whether this is the last node
      if (curr_ -> next == NULL)
      {
        break;
      }
      else
      {
        // Check whether this is the second to last node
        if ((curr_ -> next -> next) == NULL)
        {
          break;
        }
        else
        {
          // There are at least three nodes remaining in the list
          // Reverse first node
          if (after_ != NULL)
          {
            // Not at the last three nodes
            after_ -> prev = curr_;
          }
          else
          {
            // At the last three nodes
            // Reset the tail_ pointer
            tail_ = curr_;
          }
        
          curr_ -> next = after_;
          curr_ -> prev = next_;

          prev_ = curr_;
          curr_ = next_;
          next_ = next_ -> prev;

          // Reverse second
          curr_ -> next = prev_;
          curr_ = next_;

          // Reverse the third
          curr_ -> prev = before_;
          if (before_ != NULL)
          {
            // Not at the first three nodes
            before_ -> next = curr_;
          }
          else
          {
            // at the first three nodes
            // Reset the head_ pointer
            head_ = curr_;
          }
          
          // Move to the next three nodes
          curr_ = after_;
          prev_ = after_;
          if (after_ != NULL)
          {
            // Not at the last three nodes
            next_ = after_ -> next -> next;
            before_ = after_ -> prev;
            after_ = after_ -> next -> next -> next;
          }
        }
      } 
    }
  }
}


/**
 * Reverses the current List.
 */
template <typename T>
void List<T>::reverse() {
  reverse(head_, tail_);
}

/**
 * Helper function to reverse a sequence of linked memory inside a List,
 * starting at startPoint and ending at endPoint. You are responsible for
 * updating startPoint and endPoint to point to the new starting and ending
 * points of the rearranged sequence of linked memory in question.
 *
 * @param startPoint A pointer reference to the first node in the sequence
 *  to be reversed.
 * @param endPoint A pointer reference to the last node in the sequence to
 *  be reversed.
 */
template <typename T>
void List<T>::reverse(ListNode *& startPoint, ListNode *& endPoint) {
  /// @todo Graded in MP3.2
  if (startPoint == endPoint)
  {
    return;
  }
  else
  {
    ListNode *curr_ = startPoint -> next;
    ListNode *next_ = NULL;
    ListNode *before_ = startPoint -> prev;
    ListNode *after_ = endPoint -> next;

    // Modify the first node and last node to be reversed
    if (before_ != NULL)
    {
      before_ -> next = endPoint;
    }
    if (after_ != NULL)
    {
      after_ -> prev = startPoint;
    }

    startPoint -> prev = startPoint -> next;
    startPoint -> next = after_;
    endPoint -> next = endPoint -> prev;
    endPoint -> prev = before_;

    // Reverse the reamaining nodes in the range
    while (curr_ != endPoint)
    {
      next_ = curr_ -> next;
      curr_ -> next = curr_ -> prev;
      curr_ -> prev = next_;
      curr_ = next_;
    }

    // Exchange startPoint and endPoint pointers
    ListNode *temp = startPoint;
    startPoint = endPoint;
    endPoint = temp;
  }
}

/**
 * Reverses blocks of size n in the current List. You should use your
 * reverse( ListNode * &, ListNode * & ) helper function in this method!
 *
 * @param n The size of the blocks in the List to be reversed.
 */
template <typename T>
void List<T>::reverseNth(int n) {
  /// @todo Graded in MP3.2
  if (length_ < n)
  {
    // Less than n nodes in the list
    // No need to change
    return;
  }
  else
  {
    // >= n nodes in the list
    ListNode *start_ = head_; 
    ListNode *end_ = head_;
    // Locate the end_ pointer pointing to the nth nodes starting from head
    for (int i = 0; i < n - 1; i++)
    {
      end_ = end_ -> next;
    }

    while(true)
    {
      reverse(start_, end_);
      if (end_ == head_)
      {
        head_ = start_;
      }
      else if (tail_ == start_)
      {
        tail_ = end_;
        break;
      }

      // Continue reversing if not reaching the end
      // Reset start_ and end_ pointer
      start_ = end_ -> next;
      end_ = start_;
      for (int i = 0; i < n - 1 && end_ != tail_; i++)
      {
        end_ = end_ -> next;
      }  
    }
  }
}


/**
 * Merges the given sorted list into the current sorted list.
 *
 * @param otherList List to be merged into the current list.
 */
template <typename T>
void List<T>::mergeWith(List<T> & otherList) {
    // set up the current list
    head_ = merge(head_, otherList.head_);
    tail_ = head_;

    // make sure there is a node in the new list
    if (tail_ != NULL) {
        while (tail_->next != NULL)
            tail_ = tail_->next;
    }
    length_ = length_ + otherList.length_;

    // empty out the parameter list
    otherList.head_ = NULL;
    otherList.tail_ = NULL;
    otherList.length_ = 0;
}


/**
 * Helper function to merge two **sorted** and **independent** sequences of
 * linked memory. The result should be a single sequence that is itself
 * sorted.
 *
 * This function **SHOULD NOT** create **ANY** new List objects.
 *
 * @param first The starting node of the first sequence.
 * @param second The starting node of the second sequence.
 * @return The starting node of the resulting, sorted sequence.
 */

template <typename T>
typename List<T>::ListNode * List<T>::merge(ListNode * first, ListNode* second) {
  /// @todo Graded in MP3.2
  /*
  if (first == NULL && second == NULL)
  {
    return NULL;
  }
  else if (first == NULL && second != NULL)
  {
    return second;
  }
  else if (first != NULL && second == NULL)
  {
    return first;
  }
  else
  {
    // The two lists must both be nonempty
    ListNode *head_ret = first;
    ListNode *curr_fir = first;
    ListNode *curr_sec = second;
    ListNode *next_sec = second -> next;

    while (curr_sec != NULL)
    {
      next_sec = curr_sec -> next;
      curr_fir = first;
      // Trace through the first list
      for (curr_fir = first; curr_fir != NULL; curr_fir = curr_fir -> next)
      {
        if (curr_fir == first)
        {
          if (curr_sec -> data < curr_fir -> data)
          {
            // Insert at the very front of the first list
            curr_sec -> prev = NULL;
            curr_sec -> next = curr_fir;
            curr_fir -> prev = curr_sec;
            head_ret = curr_sec;
            first = curr_sec;
            break;
          }
          else 
          {
            if (curr_fir -> next != NULL)
            {
              // The first list has at least two nodes
              if (curr_sec -> data < curr_fir -> next -> data)
              {
                // Insert behind the first node of first list
                curr_sec -> next = curr_fir -> next;
                curr_fir -> next -> prev = curr_sec;
                curr_fir -> next = curr_sec;
                curr_sec -> prev = curr_fir;
                break;
              }
            }
            else
            {
              // The first list only has one node
              // Insert at the very end of the first list
              curr_sec -> prev = curr_fir;
              curr_sec -> next = NULL;
              curr_fir -> next = curr_sec;
              break;
            }
          }
        }
        else if (curr_fir -> next == NULL)
        {
          if ((curr_sec -> data < curr_fir -> data) == false)
          {
            // Insert at the very end of the first list
            curr_sec -> prev = curr_fir;
            curr_sec -> next = NULL;
            curr_fir -> next = curr_sec;
            break;
          }
        }
        else
        {
          // In the middle of the first list
          if (((curr_sec -> data < curr_fir -> data) == false) && (curr_sec -> data < curr_fir -> next -> data))
          {
            // Insert behind the current node of first list
            curr_sec -> next = curr_fir -> next;
            curr_fir -> next -> prev = curr_sec;
            curr_fir -> next = curr_sec;
            curr_sec -> prev = curr_fir;
            break;
          }
        }
      }
      curr_sec = next_sec;
    }
    return head_ret;
  }
*/

  if (first == NULL && second == NULL) 
  {
    return NULL;
  }
  if (second == NULL && first != NULL) 
  {
    return first;
  }
  if (second != NULL && first == NULL) 
  {
    return second;
  }

  // Both lists must be nonempty
  ListNode *head_ret;
  if (first -> data < second -> data) 
  {
    head_ret = first;
    first = first -> next;
  } 
  else 
  {
    head_ret = second;
    second = second -> next;
  }

  ListNode *curr_ = head_ret;
  while (first != NULL && second != NULL) 
  {
    if (first -> data < second -> data) 
    {
      curr_-> next = first;
      first -> prev = curr_;
      curr_ = first;
      first = first -> next;
    } 
    else 
    {
      curr_ -> next = second;
      second -> prev = curr_;
      curr_ = second;
      second = second -> next;
    }
  }

  // Check which iterator reaches the end first
  if (first == NULL && second != NULL) 
  {
    curr_ -> next = second;
    second -> prev = curr_;
  }
  if (second == NULL && first != NULL) 
  {
    curr_ -> next = first;
    first -> prev = curr_;
  }
  return head_ret;
}


/**
 * Sorts a chain of linked memory given a start node and a size.
 * This is the recursive helper for the Mergesort algorithm (i.e., this is
 * the divide-and-conquer step).
 *
 * Called by the public sort function in List-given.hpp
 *
 * @param start Starting point of the chain.
 * @param chainLength Size of the chain to be sorted.
 * @return A pointer to the beginning of the now sorted chain.
 */
template <typename T>
typename List<T>::ListNode* List<T>::mergesort(ListNode * start, int chainLength) {
  /// @todo Graded in MP3.2
  ListNode *first = NULL;
  ListNode *second = NULL;

  // Base case
  if (chainLength == 1)
  {
    return start;
  }
  else
  {
    // Split the sublist into two parts
    first = start;
    second = split(start, chainLength / 2);
    // sort the two parts seperately
    first = mergesort(first, chainLength / 2);
    second = mergesort(second, chainLength - chainLength / 2);
    // Combine the two parts
    first = merge(first, second);
    return first;
  }
}
