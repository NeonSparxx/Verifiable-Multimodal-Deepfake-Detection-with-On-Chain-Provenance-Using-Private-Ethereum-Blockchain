// SPDX-License-Identifier: MIT
pragma solidity ^0.8.17;

contract PredictionLogger {
    event PredictionLogged(
        string fileName,
        string prediction,
        string confidence,
        uint256 timestamp,
        address detector
    );

    function logPrediction(
        string memory fileName,
        string memory prediction,
        string memory confidence
    ) public {
        emit PredictionLogged(
            fileName,
            prediction,
            confidence,
            block.timestamp,
            msg.sender
        );
    }
}