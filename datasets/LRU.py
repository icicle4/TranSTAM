
class DoubleDirNode:
    def __init__(self, key=0, value=0):
        self.key = key
        self.value = value
        self.pre = None
        self.next = None


class LRUCache:
    def __init__(self, capacity: int):
        self.cache = {}
        self.capacity = capacity
        self.head = DoubleDirNode()
        self.tail = DoubleDirNode()
        self.head.next = self.tail
        self.tail.pre = self.head

    def get(self, key):
        if self.cache.get(key):
            self.moveToHead(self.cache[key])
            return self.cache[key].value
        else:
            return None

    def put(self, key, value) -> None:
        if self.cache.get(key):
            self.moveToHead(self.cache[key])
            self.cache[key].value = value
        else:
            if self.capacity == len(self.cache):
                new_node = DoubleDirNode(key, value)
                removed = self.removeTailNode()
                self.cache.pop(removed.key)
                self.AddToHead(new_node)
                self.cache[key] = new_node
            else:
                new_node = DoubleDirNode(key, value)
                self.AddToHead(new_node)
                self.cache[key] = new_node

    def AddToHead(self, node):
        node.next = self.head.next
        node.pre = self.head
        self.head.next.pre = node
        self.head.next = node

    def moveToHead(self, node):
        self.removeNode(node)
        self.AddToHead(node)

    def removeNode(self, node):
        node.pre.next = node.next
        node.next.pre = node.pre

    def removeTailNode(self):
        removed_node = self.tail.pre
        self.removeNode(removed_node)
        return removed_node
