from typing import List
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left=0
        right=len(nums)-1
        while left<=right:
            mid=(left+right)//2
            if nums[mid]<nums[right]:
                right=mid
            elif nums[mid]>nums[right]:
                left=mid+1
            else:
                return nums[mid]
if __name__ == "__main__":
    nums=[3,4,5,1,2]
    print(Solution().findMin(nums))