/**
 * @file avltree.cpp
 * Definitions of the binary tree functions you'll be writing for this lab.
 * You'll need to modify this file.
 */
#include <algorithm>
using namespace std;

template <class K, class V>
V AVLTree<K, V>::find(const K& key) const
{
    return find(root, key);
}

template <class K, class V>
V AVLTree<K, V>::find(Node* subtree, const K& key) const
{
    if (subtree == NULL)
        return V();
    else if (key == subtree->key)
        return subtree->value;
    else {
        if (key < subtree->key)
            return find(subtree->left, key);
        else
            return find(subtree->right, key);
    }
}


template <class K, class V>
int AVLTree<K, V>::getHeight(Node* root)
{
    if (root == NULL)
    {
        return -1;
    }

    return (1 + max(getHeight(root -> left), getHeight(root -> right)));
}

template <class K, class V>
void AVLTree<K, V>::updateHeight(Node *& cur) 
{
    cur->height = 1 + std::max(heightOrNeg1(cur->left), heightOrNeg1(cur->right));
}

template <class K, class V>
void AVLTree<K, V>::rotateLeft(Node*& t)
{
    functionCalls.push_back("rotateLeft"); // Stores the rotation name (don't remove this)
    // your code here
    Node *temp = t -> right -> left;
    Node *right_child = t -> right;
    t -> right -> left = t;   
    t -> right = temp;

    // Update height
    // t -> height = getHeight(t);
    // right_child -> height = getHeight(right_child);
    updateHeight(right_child);
    updateHeight(t);
    t = right_child;
}

template <class K, class V>
void AVLTree<K, V>::rotateLeftRight(Node*& t)
{
    functionCalls.push_back("rotateLeftRight"); // Stores the rotation name (don't remove this)
    // Implemented for you:
    rotateLeft(t->left);
    rotateRight(t);
}

template <class K, class V>
void AVLTree<K, V>::rotateRight(Node*& t)
{
    functionCalls.push_back("rotateRight"); // Stores the rotation name (don't remove this)
    // your code here
    Node *temp = t -> left -> right;
    Node *left_child = t -> left;
    t -> left -> right = t;   
    t -> left = temp;

    // Update height
    // t -> height = getHeight(t);
    // left_child -> height = getHeight(left_child);
    updateHeight(left_child);
    updateHeight(t);
    t = left_child;
}

template <class K, class V>
void AVLTree<K, V>::rotateRightLeft(Node*& t)
{
    functionCalls.push_back("rotateRightLeft"); // Stores the rotation name (don't remove this)
    // your code here
    rotateRight(t -> right);
    rotateLeft(t);
}

template <class K, class V>
void AVLTree<K, V>::rebalance(Node*& subtree)
{
    // your code here
    int balance = heightOrNeg1(subtree -> right) - heightOrNeg1(subtree -> left);

    if (balance == -2)
    {
        int left_balance = heightOrNeg1(subtree -> left -> right) - heightOrNeg1(subtree -> left -> left);
        if (left_balance == -1)
        {
            rotateRight(subtree);
        }
        else if (left_balance == 1)
        {
            rotateLeftRight(subtree);
        }
    }
    else if (balance == 2)
    {
        int right_balance = heightOrNeg1(subtree -> right -> right) - heightOrNeg1(subtree -> right -> left);
        if (right_balance == 1)
        {
            rotateLeft(subtree);
        }
        else if (right_balance == -1)
        {
            rotateRightLeft(subtree);
        }
    }
    updateHeight(subtree);

}

template <class K, class V>
void AVLTree<K, V>::insert(const K & key, const V & value)
{
    insert(root, key, value);
}

template <class K, class V>
void AVLTree<K, V>::insert(Node*& subtree, const K& key, const V& value)
{
    if (subtree == NULL)
    {
        subtree = new Node(key, value);
    }
    else if (key < subtree -> key)
    {
        insert(subtree -> left, key, value);
    }
    else if (key > subtree -> key)
    {
        insert(subtree -> right, key, value);
    }
    rebalance(subtree);
}

template <class K, class V>
void AVLTree<K, V>::remove(const K& key)
{
    remove(root, key);
}

template <class K, class V>
void AVLTree<K, V>::remove(Node*& subtree, const K& key)
{
    if (subtree == NULL)
        return;

    if (key < subtree->key) {
        // your code here
        remove(subtree -> left, key);
        rebalance(subtree);
    } else if (key > subtree->key) {
        // your code here
        remove(subtree -> right, key);
        rebalance(subtree);
    } else {
        if (subtree->left == NULL && subtree->right == NULL) {
            /* no-child remove */
            // your code here
            delete subtree;
            subtree = NULL;
        } else if (subtree->left != NULL && subtree->right != NULL) {
            /* two-child remove */
            // Find IOP
            Node *iop = subtree -> left;
            while(iop -> right != NULL)
            {
                iop = iop -> right;
            }
            swap(iop, subtree);
            remove(subtree -> left, key);
        } else {
            /* one-child remove */
            // your code here
            Node* child;
            if(subtree->left != NULL)
            {
              child = subtree->left;
            }
            else
            {
              child = subtree->right;
            }
            delete subtree;
            subtree = child;
        }
    }
}
