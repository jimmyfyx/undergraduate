/**
 * @file quackfun.cpp
 * This is where you will implement the required functions for the
 * stacks and queues portion of the lab.
 */

namespace QuackFun {

/**
 * Sums items in a stack.
 *
 * **Hint**: think recursively!
 *
 * @note You may modify the stack as long as you restore it to its original
 * values.
 *
 * @note You may use only two local variables of type T in your function.
 * Note that this function is templatized on the stack's type, so stacks of
 * objects overloading the + operator can be summed.
 *
 * @note We are using the Standard Template Library (STL) stack in this
 * problem. Its pop function works a bit differently from the stack we
 * built. Try searching for "stl stack" to learn how to use it.
 *
 * @param s A stack holding values to sum.
 * @return  The sum of all the elements in the stack, leaving the original
 *          stack in the same state (unchanged).
 */
template <typename T>
T sum(stack<T>& s)
{
    // Your code here
    // Base case
    if (s.empty() == true)
    {
        return 0;
    }

    // Recursive case
    T top = s.top();
    s.pop();
    T sum_ = top + sum(s);
    s.push(top);
    return sum_; // stub return value (0 for primitive types). Change this!
                            // Note: T() is the default value for objects, and 0 for
                            // primitive types
}

/**
 * Checks whether the given string (stored in a queue) has balanced brackets.
 * A string will consist of square bracket characters, [, ], and other
 * characters. This function will return true if and only if the square bracket
 * characters in the given string are balanced. For this to be true, all
 * brackets must be matched up correctly, with no extra, hanging, or unmatched
 * brackets. For example, the string "[hello][]" is balanced, "[[][[]a]]" is
 * balanced, "[]]" is unbalanced, "][" is unbalanced, and "))))[cs225]" is
 * balanced.
 *
 * For this function, you may only create a single local variable of type
 * `stack<char>`! No other stack or queue local objects may be declared. Note
 * that you may still declare and use other local variables of primitive types.
 *
 * @param input The queue representation of a string to check for balanced brackets in
 * @return      Whether the input string had balanced brackets
 */
bool isBalanced(queue<char> input)
{
    // @TODO: Make less optimistic
    std::stack<char> bracket;

    char first;
    unsigned int orig_size = input.size();
    for (unsigned int i = 0; i < orig_size; i++)
    {
        // Pop the first element of the queue and store it in the stack
        // Only if the element is '[' 
        first = input.front();
        // std::cout << first << std::endl;
        if (first == '[')
        {
            bracket.push(first);
        }

        if (first == ']')
        {
            // Check whether there is a left bracket in the stack
            if (bracket.empty() == true)
            {
                return false;
            }
            else
            {
                bracket.pop();
            }
        }

        input.pop();
    }

    // Check whether the bracket is empty
    if (bracket.empty() == true)
    {
        return true;
    }
    else
    {
        return false;
    }

}

/**
 * Reverses even sized blocks of items in the queue. Blocks start at size
 * one and increase for each subsequent block.
 *
 * **Hint**: You'll want to make a local stack variable.
 *
 * @note Any "leftover" numbers should be handled as if their block was
 * complete.
 *
 * @note We are using the Standard Template Library (STL) queue in this
 * problem. Its pop function works a bit differently from the stack we
 * built. Try searching for "stl stack" to learn how to use it.
 *
 * @param q A queue of items to be scrambled
 */
template <typename T>
void scramble(queue<T>& q)
{
    stack<T> s;
    // optional: queue<T> q2;
    unsigned int org_size = q.size();
    unsigned int scramble_count = 0;
    for (unsigned int i = 1; i < org_size; i++)
    {
        if (org_size == scramble_count)
        {
            break;
        }

        if (i % 2 != 0)
        {
            if (i > org_size - scramble_count)
            {
                // 'leftover' condition
                // Not change the order of the i number of elements
                unsigned int left = org_size - scramble_count;
                for (unsigned int j = 0; j < left; j++)
                {
                    T pop = q.front();
                    q.pop();
                    q.push(pop);
                }
                scramble_count = scramble_count + left;
            }
            else
            {
                // Not change the order of the i number of elements
                for (unsigned int j = 0; j < i; j++)
                {
                    T pop = q.front();
                    q.pop();
                    q.push(pop);
                }
                // std::cout << "Condition 1" << std::endl;
                scramble_count = scramble_count + i;
            }
            
        }
        else if (i % 2 == 0)
        {
            if (i > org_size - scramble_count)
            {
                // 'leftover' condition
                // Reverse the order of the 'left' number of elements
                unsigned int left = org_size - scramble_count;
                for (unsigned int j = 0; j < left; j++)
                {
                    T pop_1 = q.front();
                    q.pop();
                    s.push(pop_1);
                }
                scramble_count = scramble_count + left;
            }
            else
            {
                // Reverse the order of the i number of elements
                for (unsigned int j = 0; j < i; j++)
                {
                    T pop_1 = q.front();
                    q.pop();
                    s.push(pop_1);
                }
                scramble_count = scramble_count + i;
            }
            
            unsigned int size_ = s.size();
            for (unsigned int m = 0; m < size_; m++)
            {
                T top = s.top();
                q.push(top);
                s.pop();
            }
        }
    }
}

}
