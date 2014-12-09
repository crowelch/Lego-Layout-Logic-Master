#pragma once

#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <iostream>
#include <tchar.h>

namespace ConsoleApplication6
{
	class Program
	{

	public:
		
		static int _tmain(int argc, _TCHAR* argv[]);

		static bool inArray(std::vector<int> &block, int size);

		static bool equivalentBlockOnebyN(std::vector<int> &block, int row, int col, std::vector<std::vector<int>> &array_Renamed, bool isRowCompare);

		static bool notInQueue(std::vector<std::wstring> &queue, std::vector<int> &block, int row, int col, bool isRowCompare);

		static void addToQueue(std::vector<std::wstring> &queue, std::vector<int> &block, int row, int col, bool isRowCompare);

		static void general(std::vector<std::wstring> &queue, std::vector<std::vector<int>> &array_Renamed, int row, int col, int blR, int blC, const std::wstring &blockType);

		static void twoByTwo(std::vector<std::wstring> &queue, std::vector<std::vector<int>> &array_Renamed, int row, int col);

		static void twoByThree(std::vector<std::wstring> &queue, std::vector<std::vector<int>> &array_Renamed, int row, int col);

		static void twoByFour(std::vector<std::wstring> &queue, std::vector<std::vector<int>> &array_Renamed, int row, int col);
	};
}
