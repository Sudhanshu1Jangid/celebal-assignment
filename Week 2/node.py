class Node:
    def __init__(self, data):
        self.data = data  # Stores the value
        self.next = None  # Pointer to the next node


# LinkedList class to handle operations on the list
class LinkedList:
    def __init__(self):
        self.head = None  # Start with an empty list

    # Method to add a new node at the end of the list
    def add_node(self, data):
        new_node = Node(data)

        if self.head is None:
            self.head = new_node  # First node becomes the head
        else:
            temp = self.head
            while temp.next is not None:
                temp = temp.next
            temp.next = new_node  # Link last node to the new node

    # Method to print the entire list
    def print_list(self):
        if self.head is None:
            print("List is empty.")
        else:
            temp = self.head
            while temp is not None:
                print(temp.data, end=" -> ")
                temp = temp.next
            print("None")

    # Method to delete the nth node (1-based index)
    def delete_nth_node(self, n):
        try:
            if self.head is None:
                raise Exception("List is empty. Cannot delete.")

            if n <= 0:
                raise Exception("Index should be 1 or more.")

            if n == 1:
                # Delete the head node
                print(f"Deleted node at position {n} with value {self.head.data}")
                self.head = self.head.next
                return

            # Find the node before the one to delete
            temp = self.head
            for i in range(n - 2):
                if temp is None or temp.next is None:
                    raise Exception("Index out of range.")
                temp = temp.next

            # Check if next node exists
            if temp.next is None:
                raise Exception("Index out of range.")

            print(f"Deleted node at position {n} with value {temp.next.data}")
            temp.next = temp.next.next  # Skip the node to delete

        except Exception as e:
            print("Error:", e)


# Testing the LinkedList implementation
if __name__ == "__main__":
    # Create an object of LinkedList
    my_list = LinkedList()

    print("Adding elements:")
    my_list.add_node(5)
    my_list.add_node(10)
    my_list.add_node(15)
    my_list.add_node(20)
    my_list.print_list()

    print("\nDeleting 2nd node:")
    my_list.delete_nth_node(2)
    my_list.print_list()

    print("\nDeleting head (1st node):")
    my_list.delete_nth_node(1)
    my_list.print_list()

    print("\nTrying to delete 10th node (out of range):")
    my_list.delete_nth_node(10)

    print("\nTrying to delete all remaining nodes:")
    my_list.delete_nth_node(1)
    my_list.delete_nth_node(1)
    my_list.delete_nth_node(1)  # This should show error (list empty)
    my_list.print_list()